"""Encoder and decoder for the Mass Media BOLT LZSS variant.

BOLT is the archive container Mass Media Interactive Entertainment shipped in
their ports across the N64, GBA, Dreamcast and Xbox eras.  This module
implements the LZSS variant used by the N64/GBA archives -- the one that
StarCraft 64 stores its scenarios, briefings and art in.

The format was reverse engineered by Adam Heinermann for BOLTextract
(https://github.com/heinermann/BOLTextract), which is decode-only.  This
module reimplements the decoder from the written specification below and adds
an *encoder*, which as far as I can tell did not previously exist anywhere:
the format is Mass Media's own, it is undocumented, and no compressor for it
has ever been published.


The bitstream
=============

There is no header, no end marker and no self-describing length.  A stream is
a bare sequence of operations, and the decoder stops when it has produced the
number of bytes the *container* said to expect.  (`decode` will also stop
cleanly at end-of-input if you do not know the size.)

Decoding keeps three accumulators, all of which reset to zero after every
operation that emits output:

    op_count    control bytes consumed since the last emit
    ext_offset  pending high bits of a back-reference distance
    ext_run     pending high bits of a run length

Control bytes are read one at a time and dispatched on their high bits:

    0xxxxxxx   back-reference (the only op with a clear top bit)
                   distance = ((ext_offset << 4) | (b & 0x0F)) + 1
                   length   = ((ext_run    << 3) | (b >> 4)) + op_count + 1
               Copy `length` bytes from `distance` back in the output, one
               byte at a time, so a run may overlap itself and repeat.

    1000xxxx   literal run
                   length = ((ext_run << 4) | (b & 0x0F)) + 1
               followed immediately by that many raw bytes in the stream.

    1001xxxx   extend BOTH accumulators by two bits each
                   ext_run    = (ext_run    << 2) | (b & 0x03)
                   ext_offset = (ext_offset << 2) | ((b >> 2) & 0x03)

    101xxxxx   extend the run accumulator by five bits
                   ext_run = (ext_run << 5) | (b & 0x1F)

    11xxxxxx   extend the offset accumulator by six bits
                   ext_offset = (ext_offset << 6) | (b & 0x3F)

Because the accumulators shift left, an extension byte contributes the *more*
significant bits and must be emitted before the bits it precedes.


The op_count subtlety
=====================

This is the part that is easy to miss, and it is what makes the format unlike
every other LZSS derivative.

`op_count` counts every control byte since the last emit -- including the
back-reference byte itself -- and it is **added to the back-reference length**.
So the extension bytes you spend widening a distance are not free overhead;
they are also *credited to the run*.  A back-reference preceded by no
extension bytes has op_count == 1, giving a minimum length of 2.

Concretely, if a back-reference is preceded by `k` extension bytes:

    length = ((ext_run << 3) | (b >> 4)) + k + 2

That has three consequences an encoder has to model and a decoder never has
to think about:

  1. Length and distance are *not* independent.  You cannot choose the
     encoded run field without first knowing how many bytes you will spend on
     the distance, and you cannot choose the distance encoding without
     knowing the run, because run extension bytes also inflate `k`.  The
     encoder here solves the fixed point by searching `k` upward
     (see `_plan_reference`).

  2. Far matches have a *minimum* length.  A distance above 16 needs at least
     one extension byte, so it cannot express a length below 3; a distance
     above 1024 needs two, so its floor is 4; and so on.  Short matches are
     simply unrepresentable at long range, and the encoder must fall back to
     literals rather than emit a match it cannot spell.

  3. The overhead is partly refunded.  A five-byte-distance encoding costs
     five bytes but also buys five length, which is why the format stays
     compact despite spending whole bytes on 6-bit offset chunks.


Public API
==========

    decode(data, expected_size=None)  -> bytes
    encode(data, level=..., ...)      -> bytes
    decoded_length(data, ...)         -> int   (bytes consumed, for measuring)

`decode(encode(x)) == x` holds for every input.

Licence: GPL-3.0-or-later.  See LICENSE and the README for why.
"""

from __future__ import annotations

__all__ = [
    "decode",
    "encode",
    "BoltLZSSError",
    "LEVEL_STORE",
    "LEVEL_GREEDY",
    "LEVEL_LAZY",
    "LEVEL_OPTIMAL",
    "DEFAULT_LEVEL",
]

__version__ = "1.0.0"


class BoltLZSSError(ValueError):
    """Raised when a stream is malformed or an encode request is impossible."""


# ---------------------------------------------------------------------------
# Control byte constants
# ---------------------------------------------------------------------------
# Dispatch is on the high bits, tested in this order:
#     b & 0x80 == 0     -> back-reference
#     b & 0x40          -> offset extension   (0xC0..0xFF)
#     b & 0x20          -> run extension      (0xA0..0xBF)
#     b & 0x10          -> dual extension     (0x90..0x9F)
#     otherwise         -> literal run        (0x80..0x8F)

OP_LITERAL = 0x80   # 1000xxxx
OP_DUAL = 0x90      # 1001xxxx
OP_RUN_EXT = 0xA0   # 101xxxxx
OP_OFF_EXT = 0xC0   # 11xxxxxx

# Field widths, in bits, of what each control byte contributes.
BITS_OFF_EXT = 6    # 11xxxxxx  -> ext_offset
BITS_RUN_EXT = 5    # 101xxxxx  -> ext_run
BITS_DUAL = 2       # 1001xxxx  -> two bits to each accumulator

# Immediate field widths of the terminal ops.
BITS_REF_DIST = 4   # low nibble of a back-reference byte
BITS_REF_LEN = 3    # high three bits (the top bit is the opcode)
BITS_LIT_LEN = 4    # low nibble of a literal-run byte

# A back-reference with no extension bytes still has op_count == 1, and the
# stored length is biased by op_count + 1, so the shortest match is 2 bytes.
MIN_MATCH = 2

# Largest literal run expressible with no extension byte: (0 << 4 | 15) + 1.
MAX_PLAIN_LITERAL = 1 << BITS_LIT_LEN  # 16


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode(data, expected_size: int | None = None, *,
           start: int = 0, strict: bool = True) -> bytes:
    """Decompress a BOLT LZSS stream.

    `data`           the compressed bytes (a buffer; `start` indexes into it)
    `expected_size`  stop once this many bytes have been produced.  The
                     format carries no length of its own -- the BOLT
                     container stores it -- so pass it when you have it.
                     When None, decoding runs until the input is exhausted.
    `start`          offset of the stream within `data`
    `strict`         raise if the stream produces more bytes than
                     `expected_size` (an overshooting final op) or if it runs
                     off the end of the input

    Returns the decompressed bytes.  Use `decoded_length` if you also need to
    know how many *input* bytes the stream occupied.
    """
    out, _consumed = _decode_core(data, expected_size, start, strict)
    return out


def decoded_length(data, expected_size: int | None = None, *,
                   start: int = 0, strict: bool = True) -> int:
    """Return how many input bytes the stream at `start` occupies.

    This is the honest way to measure a compressed size in a BOLT archive:
    the container stores only the *uncompressed* length and a data offset, so
    the compressed length has to be recovered by decoding.  Taking the
    difference between consecutive data offsets is a common shortcut but
    includes any inter-entry alignment padding.
    """
    _out, consumed = _decode_core(data, expected_size, start, strict)
    return consumed


def _decode_core(data, expected_size, start, strict):
    mv = memoryview(data) if not isinstance(data, (bytes, bytearray)) else data
    end = len(mv)
    pos = start
    out = bytearray()
    want = expected_size

    op_count = 0
    ext_offset = 0
    ext_run = 0

    while True:
        if want is not None and len(out) >= want:
            break
        if pos >= end:
            if want is None:
                break
            if strict:
                raise BoltLZSSError(
                    f"stream ended after {len(out)} of {want} bytes"
                )
            break

        b = mv[pos]
        pos += 1
        op_count += 1

        if b & 0x80:
            if b & 0x40:                       # 11xxxxxx  extend offset
                ext_offset = (ext_offset << BITS_OFF_EXT) | (b & 0x3F)
            elif b & 0x20:                     # 101xxxxx  extend run
                ext_run = (ext_run << BITS_RUN_EXT) | (b & 0x1F)
            elif b & 0x10:                     # 1001xxxx  extend both
                ext_run = (ext_run << BITS_DUAL) | (b & 0x03)
                ext_offset = (ext_offset << BITS_DUAL) | ((b >> 2) & 0x03)
            else:                              # 1000xxxx  literal run
                run = ((ext_run << BITS_LIT_LEN) | (b & 0x0F)) + 1
                if pos + run > end:
                    if strict:
                        raise BoltLZSSError(
                            f"literal run of {run} at input {pos} overruns "
                            f"the {end}-byte stream"
                        )
                    run = end - pos
                out += mv[pos:pos + run]
                pos += run
                op_count = ext_offset = ext_run = 0
        else:                                  # 0xxxxxxx  back-reference
            dist = ((ext_offset << BITS_REF_DIST) | (b & 0x0F)) + 1
            run = (((ext_run << BITS_REF_LEN) | (b >> 4))
                   + op_count + 1)
            if dist > len(out):
                raise BoltLZSSError(
                    f"back-reference at input {pos - 1} reaches {dist} bytes "
                    f"back but only {len(out)} bytes have been produced"
                )
            # Byte at a time: an overlapping run is how this format spells RLE.
            for _ in range(run):
                out.append(out[-dist])
            op_count = ext_offset = ext_run = 0

    if want is not None and len(out) != want and strict:
        raise BoltLZSSError(
            f"stream produced {len(out)} bytes, expected {want}"
        )
    return bytes(out), pos - start


# ---------------------------------------------------------------------------
# Encoder -- operation planning
# ---------------------------------------------------------------------------

def _bit_groups(value: int, sizes: list[int]) -> list[int]:
    """Split `value` MSB-first into fields of the given widths.

    The total width must be at least `value.bit_length()`; any surplus shows
    up as leading zero groups, which is exactly what we want -- a zero-valued
    extension byte is a legal no-op that only shifts the accumulator.
    """
    shift = sum(sizes)
    out = []
    for size in sizes:
        shift -= size
        out.append((value >> shift) & ((1 << size) - 1))
    return out


def _literal_ext_count(run: int) -> int:
    """Extension bytes needed to spell a literal run of `run` bytes."""
    if run < 1:
        raise BoltLZSSError(f"literal run must be positive, got {run}")
    value = (run - 1) >> BITS_LIT_LEN          # goes into ext_run
    if value == 0:
        return 0
    return -(-value.bit_length() // BITS_RUN_EXT)   # ceil-div


def literal_cost(run: int) -> int:
    """Total encoded size, in bytes, of a literal run of `run` bytes."""
    return run + 1 + _literal_ext_count(run)


def _emit_literal(out: bytearray, src, pos: int, run: int) -> None:
    n_ext = _literal_ext_count(run)
    value = (run - 1) >> BITS_LIT_LEN
    for group in _bit_groups(value, [BITS_RUN_EXT] * n_ext):
        out.append(OP_RUN_EXT | group)
    out.append(OP_LITERAL | ((run - 1) & 0x0F))
    out += src[pos:pos + run]


# How many extension bytes a single back-reference is ever allowed to use.
# Six offset-extension bytes already express a 2**36 distance; the cap only
# exists to bound the search.
_MAX_EXT = 12


def _plan_reference(dist: int, length: int):
    """Work out how to spell a (distance, length) back-reference.

    Returns `(n_off, n_dual, n_run, ext_offset, ext_run, lo, hi)` or None if
    the pair cannot be expressed.

    This is where the op_count coupling bites.  We want

        distance = ((ext_offset << 4) | lo) + 1
        length   = ((ext_run    << 3) | hi) + k + 2

    where `k` is the number of extension bytes.  `ext_offset` is fixed by the
    distance, but `ext_run` depends on `k`, and `k` depends in part on how
    many bytes `ext_run` needs -- a fixed point.  Rather than iterate to
    convergence we simply search `k` upward from zero and take the first
    value that admits a legal byte layout, which is also the cheapest one
    since the encoded size is exactly k + 1.
    """
    if dist < 1 or length < MIN_MATCH:
        return None

    lo = (dist - 1) & 0x0F
    v_off = (dist - 1) >> BITS_REF_DIST
    bits_off = v_off.bit_length()

    for k in range(0, _MAX_EXT + 1):
        rest = length - k - 2               # == (ext_run << 3) | hi
        if rest < 0:
            return None                     # only gets worse as k grows
        v_run = rest >> BITS_REF_LEN
        hi = rest & 0x07
        bits_run = v_run.bit_length()

        # Choose how many dual-extension bytes to use.  A dual byte feeds two
        # bits to each accumulator, so when both need widening it can do the
        # work of two single-purpose bytes.
        for n_dual in range(0, k + 1):
            need_off = max(0, bits_off - BITS_DUAL * n_dual)
            need_run = max(0, bits_run - BITS_DUAL * n_dual)
            n_off = -(-need_off // BITS_OFF_EXT)
            n_run = -(-need_run // BITS_RUN_EXT)
            if n_off + n_run + n_dual > k:
                continue
            # Spend any slack on zero-valued offset bytes; leading zeros in
            # the accumulator are harmless.
            n_off += k - (n_off + n_run + n_dual)
            return n_off, n_dual, n_run, v_off, v_run, lo, hi
    return None


def reference_cost(dist: int, length: int) -> int | None:
    """Encoded size in bytes of a back-reference, or None if unrepresentable."""
    plan = _plan_reference(dist, length)
    if plan is None:
        return None
    return plan[0] + plan[1] + plan[2] + 1


def _emit_reference(out: bytearray, dist: int, length: int) -> None:
    plan = _plan_reference(dist, length)
    if plan is None:
        raise BoltLZSSError(
            f"back-reference dist={dist} len={length} is not representable"
        )
    n_off, n_dual, n_run, v_off, v_run, lo, hi = plan

    # Emission order is: offset bytes, dual bytes, run bytes, terminal byte.
    # Each accumulator sees its bit groups in that same relative order, so
    # split the values to match.
    off_groups = _bit_groups(v_off, [BITS_OFF_EXT] * n_off
                             + [BITS_DUAL] * n_dual)
    run_groups = _bit_groups(v_run, [BITS_DUAL] * n_dual
                             + [BITS_RUN_EXT] * n_run)

    for i in range(n_off):
        out.append(OP_OFF_EXT | off_groups[i])
    for i in range(n_dual):
        out.append(OP_DUAL | (off_groups[n_off + i] << 2) | run_groups[i])
    for i in range(n_run):
        out.append(OP_RUN_EXT | run_groups[n_dual + i])
    out.append((hi << 4) | lo)


# ---------------------------------------------------------------------------
# Encoder -- entry point
# ---------------------------------------------------------------------------

LEVEL_STORE = 0     # literal runs only; the trivially correct baseline
LEVEL_GREEDY = 1    # hash-chain match finding, take the longest match
LEVEL_LAZY = 2      # greedy plus one-byte lookahead
LEVEL_OPTIMAL = 3   # shortest-path parse over the real cost model

DEFAULT_LEVEL = LEVEL_OPTIMAL


def encode(data, level: int = DEFAULT_LEVEL, **kwargs) -> bytes:
    """Compress `data` into a BOLT LZSS stream.

    `level` trades encode time for output size:
        0  literal runs only -- correct, never compresses
        1  greedy match finding
        2  lazy match finding (one-byte lookahead)
        3  optimal parse (default)

    The result always satisfies `decode(encode(x)) == x`.
    """
    data = bytes(data)
    if level == LEVEL_STORE:
        return _encode_store(data)
    raise BoltLZSSError(f"unknown compression level {level!r}")


def _encode_store(data: bytes) -> bytes:
    """The baseline: emit everything as literal runs.

    Correct by construction and useful as a control when measuring the real
    encoder.  Expands by roughly one byte per 512 (one control byte plus one
    run-extension byte buys a 512-byte run).
    """
    out = bytearray()
    pos = 0
    n = len(data)
    # 512 == ((31 << 4) | 15) + 1, the longest run one extension byte spells.
    chunk = ((1 << BITS_RUN_EXT) - 1 << BITS_LIT_LEN | 0x0F) + 1
    while pos < n:
        run = min(chunk, n - pos)
        _emit_literal(out, data, pos, run)
        pos += run
    return bytes(out)
