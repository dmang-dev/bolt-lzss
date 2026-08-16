"""Minimal reader for the BOLT archive container.

`bolt_lzss` is the codec and knows nothing about where a stream came from.
This module is the small amount of container knowledge needed to point the
codec at real data: enough to walk a BOLT directory tree inside an N64
cartridge dump and hand back (compressed bytes, uncompressed size) pairs.

It exists so the test suite and the benchmark can be run against ground
truth.  It is stdlib-only and read-only, and it never writes anything.

Container layout, all big endian, all offsets relative to the 'BOLT' magic:

    header, 16 bytes
        'BOLT'
        u8 hour, u8 minute, u8 second, u8 sub-second   build stamp
        u8 month, u8 day, u8 year-1900
        u8 entry count (0 means 256)
        u32 purpose unknown

    entry, 16 bytes
        u8  flags          bit 0x08 = stored uncompressed
        u8  unknown
        u8  unknown
        u8  file_type      for a directory, the child count
        u32 uncompressed_size
        u32 data_offset
        u32 file_hash      zero marks a directory entry

A directory's data_offset points at its child entry array.

No game data ships with this repository.  Point `open_rom` at your own dump.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

__all__ = ["load_rom", "BoltEntry", "BoltArchive", "FLAG_UNCOMPRESSED"]

Z64_MAGIC = bytes.fromhex("80371240")   # big endian, native
V64_MAGIC = bytes.fromhex("37804012")   # 16-bit byte-swapped
N64_MAGIC = bytes.fromhex("40123780")   # 32-bit little endian

HEADER_SIZE = 16
ENTRY_SIZE = 16
FLAG_UNCOMPRESSED = 0x08


def load_rom(path: str) -> bytes:
    """Read an N64 dump and normalise it to z64 (big-endian) order.

    The extension is ignored; dumps are routinely mislabelled.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 0x40:
        raise ValueError(f"{path}: too small to be an N64 ROM")

    magic = raw[:4]
    if magic == Z64_MAGIC:
        return raw
    if magic == V64_MAGIC:
        buf = bytearray(raw)
        buf[0::2], buf[1::2] = raw[1::2], raw[0::2]
        return bytes(buf)
    if magic == N64_MAGIC:
        buf = bytearray(len(raw))
        buf[0::4], buf[1::4], buf[2::4], buf[3::4] = (
            raw[3::4], raw[2::4], raw[1::4], raw[0::4])
        return bytes(buf)
    raise ValueError(f"{path}: not an N64 ROM (header {magic.hex()})")


@dataclass(frozen=True)
class BoltEntry:
    path: str
    flags: int
    file_type: int
    size: int          # uncompressed size
    offset: int        # data offset, relative to the BOLT magic
    file_hash: int

    @property
    def stored(self) -> bool:
        """True if the payload is not compressed."""
        return bool(self.flags & FLAG_UNCOMPRESSED)


class BoltArchive:
    """Read-only view of the BOLT archive embedded in a ROM image."""

    def __init__(self, rom: bytes):
        self.rom = rom
        base = rom.find(b"BOLT")
        if base < 0:
            raise ValueError("no BOLT archive found in this image")
        self.base = base
        hdr = rom[base:base + HEADER_SIZE]
        (self.hour, self.minute, self.second, self.millis,
         self.month, self.day, year, self.num_entries) = hdr[4:12]
        self.year = 1900 + year
        self.header_u32 = struct.unpack_from(">I", hdr, 12)[0]

    @property
    def build_stamp(self) -> str:
        return (f"{self.year:04d}-{self.month:02d}-{self.day:02d} "
                f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}")

    def entries(self):
        """Yield every leaf entry, depth first."""
        yield from self._walk("", HEADER_SIZE, self.num_entries or 256, 0)

    def _walk(self, prefix: str, table: int, count: int, depth: int):
        if depth > 8:                      # corruption guard; real trees are 2
            return
        for i in range(count or 256):
            e = self._entry(f"{prefix}{i:03X}", table + i * ENTRY_SIZE)
            if e.file_hash == 0:
                yield from self._walk(e.path + "/", e.offset, e.file_type,
                                      depth + 1)
            else:
                yield e

    def _entry(self, path: str, offset: int) -> BoltEntry:
        b = self.rom[self.base + offset:self.base + offset + ENTRY_SIZE]
        size, data_off, file_hash = struct.unpack_from(">III", b, 4)
        return BoltEntry(path, b[0], b[3], size, data_off, file_hash)

    def raw(self, entry: BoltEntry) -> memoryview:
        """The entry's bytes as they sit in the ROM, still compressed."""
        return memoryview(self.rom)[self.base + entry.offset:]

    def read(self, entry: BoltEntry) -> bytes:
        """Decompress (or copy) an entry using this project's own decoder."""
        import bolt_lzss
        if entry.stored:
            start = self.base + entry.offset
            return self.rom[start:start + entry.size]
        return bolt_lzss.decode(self.rom, entry.size,
                                start=self.base + entry.offset)

    def compressed_length(self, entry: BoltEntry) -> int:
        """Exact stored size of an entry, recovered by decoding it.

        The container records only the uncompressed size, so this is the only
        way to get the real figure.  Differencing consecutive data offsets is
        the usual shortcut but folds in inter-entry alignment padding.
        """
        import bolt_lzss
        if entry.stored:
            return entry.size
        return bolt_lzss.decoded_length(self.rom, entry.size,
                                        start=self.base + entry.offset)
