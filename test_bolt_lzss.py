"""Test suite for the BOLT LZSS codec.

Run with:

    py -3.13 -m unittest

Everything that needs game data is skipped cleanly when the data is absent,
so a bare checkout passes on a machine that has never seen a cartridge dump.
To exercise the real-data tests, point the environment at your own ROM:

    BOLT_ROM=/path/to/StarCraft64.n64            (or --rom PATH in argv)
    SC64_MAPS=/path/to/sc64-maps                 (for the differential test)

No game data is committed here and none is written out.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bolt_lzss as B

# Every compression level that must satisfy decode(encode(x)) == x.
LEVELS = B.ALL_LEVELS


# ---------------------------------------------------------------------------
# Locating optional game data
# ---------------------------------------------------------------------------

def _rom_path() -> str | None:
    path = os.environ.get("BOLT_ROM")
    if not path:
        argv = sys.argv
        if "--rom" in argv:
            i = argv.index("--rom")
            if i + 1 < len(argv):
                path = argv[i + 1]
    if path and os.path.isfile(path):
        return path
    return None


_ROM_CACHE: dict[str, object] = {}


def _archive():
    """Return a BoltArchive for the configured ROM, or None."""
    if "archive" in _ROM_CACHE:
        return _ROM_CACHE["archive"]
    path = _rom_path()
    archive = None
    if path:
        import bolt_archive
        try:
            archive = bolt_archive.BoltArchive(bolt_archive.load_rom(path))
        except ValueError:
            archive = None
    _ROM_CACHE["archive"] = archive
    return archive


def _sc64_decoder():
    """Return sc64-maps' proven decoder as a callable, or None."""
    if "sc64" in _ROM_CACHE:
        return _ROM_CACHE["sc64"]
    root = os.environ.get("SC64_MAPS")
    fn = None
    if root and os.path.isdir(root):
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import extract_sc64_maps as ref
            fn = ref.BoltArchive
        except Exception:
            fn = None
    _ROM_CACHE["sc64"] = fn
    return fn


def _sample_entries(limit: int, seed: int = 0, compressed_only: bool = True):
    """A deterministic spread of entries across the archive."""
    archive = _archive()
    if archive is None:
        return []
    entries = [e for e in archive.entries()
               if not compressed_only or not e.stored]
    if len(entries) <= limit:
        return entries
    rng = random.Random(seed)
    return rng.sample(entries, limit)


# ---------------------------------------------------------------------------
# Bitstream semantics
# ---------------------------------------------------------------------------

class TestControlBytes(unittest.TestCase):
    """Hand-built streams pinning each control byte's documented meaning."""

    def test_literal_run_of_one(self):
        # 0x80 == literal run, length (0 << 4 | 0) + 1 == 1
        self.assertEqual(B.decode(b"\x80Z", 1), b"Z")

    def test_literal_run_max_without_extension(self):
        # 0x8F == length 16, the most a bare literal byte can spell
        payload = bytes(range(16))
        self.assertEqual(B.decode(b"\x8f" + payload, 16), payload)

    def test_run_extension_widens_literal_run(self):
        # 0xA1 sets ext_run = 1, so 0x80 means ((1 << 4) | 0) + 1 == 17
        payload = bytes(range(17))
        self.assertEqual(B.decode(b"\xa1\x80" + payload, 17), payload)

    def test_backreference_minimum_length_is_two(self):
        # 0x00: dist = (0 & 0xF) + 1 = 1, len = (0 >> 4) + op_count + 1
        # op_count is 1 (the reference byte itself), so len == 2.
        self.assertEqual(B.decode(b"\x80A\x00", 3), b"AAA")

    def test_backreference_length_field(self):
        # 0x70: len = 7 + 1 + 1 = 9, dist = 1  -> nine more copies
        self.assertEqual(B.decode(b"\x80A\x70", 10), b"A" * 10)

    def test_op_count_credits_extension_bytes_to_the_length(self):
        # Two identical references, one preceded by a no-op offset extension.
        # The extension byte sets ext_offset = 0 (no change to the distance)
        # but raises op_count from 1 to 2, so the run grows by exactly one.
        short = B.decode(b"\x80A\x00", 3)
        longer = B.decode(b"\x80A\xc0\x00", 4)
        self.assertEqual(short, b"AAA")
        self.assertEqual(longer, b"AAAA")

    def test_offset_extension_six_bits(self):
        # ext_offset = 1 -> dist = ((1 << 4) | lo) + 1
        data = bytes(range(20))
        # ext_run 1 then 0x83 -> literal run ((1 << 4) | 3) + 1 == 20
        stream = bytearray(b"\xa1\x83" + data)
        stream += b"\xc1\x03"                   # dist (1<<4|3)+1 = 20, len 2+1
        out = B.decode(bytes(stream), 23)
        self.assertEqual(out[:20], data)
        self.assertEqual(out[20:], data[:3])

    def test_dual_extension_feeds_both_accumulators(self):
        # 0x9F -> ext_offset |= 3, ext_run |= 3
        self.assertEqual((0x9F >> 2) & 0x03, 3)
        self.assertEqual(0x9F & 0x03, 3)
        data = bytes(range(60))
        stream = bytearray(b"\xa3\x8b" + data)  # ext_run 3 -> len (3<<4|11)+1
        self.assertEqual(((3 << 4) | 0x0B) + 1, 60)
        # dist = ((3 << 4) | 11) + 1 = 60, len = ((3 << 3) | 0) + 2 + 1 = 27
        stream += b"\x9f\x0b"
        out = B.decode(bytes(stream), 60 + 27)
        self.assertEqual(out[:60], data)
        self.assertEqual(out[60:], data[:27])

    def test_overlapping_run_is_how_the_format_spells_rle(self):
        # distance 1 with a long run repeats the last byte
        self.assertEqual(B.decode(b"\x80Q\x70", 10), b"Q" * 10)
        # distance 2 alternates
        self.assertEqual(B.decode(b"\x81AB\x71", 11), b"AB" * 5 + b"A")

    def test_backreference_before_any_output_is_rejected(self):
        with self.assertRaises(B.BoltLZSSError):
            B.decode(b"\x00", 2)

    def test_truncated_stream_is_rejected(self):
        with self.assertRaises(B.BoltLZSSError):
            B.decode(b"\x8f" + b"AB", 16)

    def test_decode_without_expected_size_runs_to_end_of_input(self):
        self.assertEqual(B.decode(b"\x80A\x70"), b"A" * 10)

    def test_decoded_length_reports_input_consumed(self):
        stream = b"\x80A\x70" + b"\xff\xff\xff"   # trailing junk
        self.assertEqual(B.decode(stream, 10), b"A" * 10)
        self.assertEqual(B.decoded_length(stream, 10), 3)


# ---------------------------------------------------------------------------
# Reference planning: the op_count fixed point
# ---------------------------------------------------------------------------

class TestReferencePlanning(unittest.TestCase):

    def _roundtrip_reference(self, dist, length):
        """Emit prefix + one reference, decode it, return the copied tail."""
        prefix = bytes((i * 37 + 11) & 0xFF for i in range(dist))
        out = bytearray()
        B._emit_literal(out, prefix, 0, len(prefix))
        before = len(out)
        B._emit_reference(out, dist, length)
        self.assertEqual(len(out) - before, B.reference_cost(dist, length))
        got = B.decode(bytes(out), len(prefix) + length)
        self.assertEqual(got[:dist], prefix)
        return got[dist:]

    def _expected_tail(self, prefix, dist, length):
        buf = bytearray(prefix)
        for _ in range(length):
            buf.append(buf[-dist])
        return bytes(buf[len(prefix):])

    def test_near_matches_cost_one_byte(self):
        # distance <= 16 and length in 2..9 needs no extension byte at all
        for dist in range(1, 17):
            for length in range(2, 10):
                self.assertEqual(B.reference_cost(dist, length), 1,
                                 f"dist={dist} len={length}")

    def test_far_matches_have_a_length_floor(self):
        # A distance above 16 needs an offset-extension byte, which raises
        # op_count, which raises the minimum expressible length to 3.
        self.assertIsNone(B.reference_cost(17, 2))
        self.assertIsNotNone(B.reference_cost(17, 3))
        # Above 1024 two extension bytes are needed, so the floor becomes 4.
        self.assertIsNone(B.reference_cost(2000, 3))
        self.assertIsNotNone(B.reference_cost(2000, 4))

    def test_exhaustive_small_space(self):
        for dist in range(1, 130):
            for length in range(2, 60):
                if B.reference_cost(dist, length) is None:
                    continue
                prefix = bytes((i * 37 + 11) & 0xFF for i in range(dist))
                tail = self._roundtrip_reference(dist, length)
                self.assertEqual(
                    tail, self._expected_tail(prefix, dist, length),
                    f"dist={dist} len={length}")

    def test_random_large_space(self):
        rng = random.Random(20240607)
        checked = 0
        for _ in range(3000):
            dist = rng.choice([1, 2, 15, 16, 17, 1023, 1024, 1025,
                               rng.randint(1, 200000)])
            length = rng.randint(2, 5000)
            if B.reference_cost(dist, length) is None:
                continue
            prefix = bytes((i * 37 + 11) & 0xFF for i in range(dist))
            tail = self._roundtrip_reference(dist, length)
            self.assertEqual(tail,
                             self._expected_tail(prefix, dist, length),
                             f"dist={dist} len={length}")
            checked += 1
        self.assertGreater(checked, 1000)

    def test_every_representable_reference_at_least_breaks_even(self):
        """A back-reference can never make the stream bigger.

        This falls straight out of the op_count bias and is the format's
        quietest design decision.  A reference preceded by k extension bytes
        costs k + 1 bytes, and its shortest expressible length is k + 2,
        because op_count credits every one of those bytes back to the run.
        So the worst a legal match can do is save one byte against spelling
        the same span out as literals.
        """
        for dist in list(range(1, 200)) + [1000, 1024, 4096, 65536, 1 << 20]:
            for length in range(2, 300):
                cost = B.reference_cost(dist, length)
                if cost is None:
                    continue
                self.assertLessEqual(cost, length - 1,
                                     f"dist={dist} len={length} cost={cost}")

    def test_length_floor_matches_the_distance_class(self):
        """Each extension byte a distance needs raises the minimum length."""
        # (largest distance needing k extension bytes, resulting floor)
        for max_dist, floor in ((16, 2), (1024, 3), (65536, 4),
                                (1 << 22, 5)):
            self.assertIsNotNone(B.reference_cost(max_dist, floor),
                                 f"dist={max_dist} len={floor}")
            if floor > 2:
                self.assertIsNone(B.reference_cost(max_dist, floor - 1),
                                  f"dist={max_dist} len={floor - 1}")

    def test_cost_never_exceeds_the_extension_budget(self):
        # Nothing in a 32 MiB address space should need more than a handful
        # of extension bytes.
        for dist in (1, 16, 1024, 65536, 1 << 20, 1 << 24):
            cost = B.reference_cost(dist, 10000)
            self.assertIsNotNone(cost)
            self.assertLessEqual(cost, 8, f"dist={dist}")


# ---------------------------------------------------------------------------
# Round-trip on synthetic data
# ---------------------------------------------------------------------------

def _corpus():
    """Synthetic inputs covering the size and entropy extremes."""
    rng = random.Random(1234567)
    out = [
        (b"", "empty"),
        (b"\x00", "one zero byte"),
        (b"A", "one byte"),
        (b"\xff" * 2, "two identical"),
        (bytes(range(256)), "every byte value"),
    ]
    # sizes around every field boundary in the format
    for n in (1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128, 129,
              255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096, 65537):
        out.append((bytes(rng.getrandbits(8) for _ in range(n)),
                    f"random {n}"))
        out.append((b"\x5a" * n, f"identical {n}"))
    # varying entropy: a small alphabet compresses, a large one does not
    for alpha in (2, 4, 16, 64, 200):
        pool = bytes(range(alpha))
        out.append((bytes(rng.choice(pool) for _ in range(8000)),
                    f"alphabet {alpha}"))
    # highly repetitive
    out.append((b"abcdefgh" * 1000, "short period"))
    out.append((b"the quick brown fox " * 500, "phrase repeat"))
    out.append(((b"x" * 300 + b"y" * 300) * 20, "long runs"))
    # long-distance repeats: a unique block, filler, then the block again
    block = bytes(rng.getrandbits(8) for _ in range(2000))
    filler = bytes(rng.getrandbits(8) for _ in range(50000))
    out.append((block + filler + block, "far repeat 50k"))
    out.append((block + filler + filler + block, "far repeat 100k"))
    # structured data, which is what the archive actually holds
    out.append((b"".join(bytes([i & 0xFF, 0, 0, 0]) for i in range(4000)),
                "sparse records"))
    return out


class TestRoundTrip(unittest.TestCase):

    def test_corpus(self):
        for level in LEVELS:
            for data, name in _corpus():
                with self.subTest(level=level, case=name, size=len(data)):
                    blob = B.encode(data, level=level)
                    self.assertEqual(B.decode(blob, len(data)), data)
                    self.assertEqual(B.decoded_length(blob, len(data)),
                                     len(blob))

    def test_decode_without_expected_size(self):
        for level in LEVELS:
            for data, name in _corpus():
                if not data:
                    continue
                with self.subTest(level=level, case=name):
                    self.assertEqual(B.decode(B.encode(data, level=level)),
                                     data)

    def test_random_fuzz(self):
        rng = random.Random(987654321)
        for level in LEVELS:
            for _ in range(150):
                n = rng.randint(0, 3000)
                # bias the alphabet so some cases are compressible
                alpha = rng.choice([2, 3, 8, 64, 256])
                data = bytes(rng.randrange(alpha) for _ in range(n))
                with self.subTest(level=level, size=n, alphabet=alpha):
                    self.assertEqual(
                        B.decode(B.encode(data, level=level), n), data)

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(B.BoltLZSSError):
            B.encode(b"abc", level=99)


class TestStoreBaseline(unittest.TestCase):
    """The literal-only encoder is the control: correct, never compressing."""

    def test_expansion_is_bounded(self):
        for n in (1, 100, 512, 5000, 100000):
            data = bytes(range(256)) * (n // 256) + bytes(n % 256)
            data = data[:n]
            blob = B.encode(data, level=B.LEVEL_STORE)
            self.assertEqual(B.decode(blob, n), data)
            # one control byte plus one extension byte buys a 512-byte run
            self.assertLessEqual(len(blob), n + 2 * (n // 512 + 1))


class TestCompressionQuality(unittest.TestCase):
    """Not correctness, but properties a sane encoder has to satisfy."""

    def test_optimal_never_loses_to_the_baseline(self):
        for data, name in _corpus():
            with self.subTest(case=name):
                self.assertLessEqual(len(B.encode(data, B.LEVEL_OPTIMAL)),
                                     len(B.encode(data, B.LEVEL_STORE)))

    def test_higher_levels_are_not_worse_on_aggregate(self):
        totals = {level: 0 for level in LEVELS}
        for data, _name in _corpus():
            for level in LEVELS:
                totals[level] += len(B.encode(data, level=level))
        self.assertLess(totals[B.LEVEL_GREEDY], totals[B.LEVEL_STORE])
        self.assertLessEqual(totals[B.LEVEL_LAZY], totals[B.LEVEL_GREEDY])
        self.assertLess(totals[B.LEVEL_OPTIMAL], totals[B.LEVEL_LAZY])

    def test_incompressible_data_barely_expands(self):
        rng = random.Random(4242)
        data = bytes(rng.getrandbits(8) for _ in range(200000))
        blob = B.encode(data, B.LEVEL_OPTIMAL)
        # a single literal run costs the payload plus a control byte plus a
        # handful of extension bytes; random data will find a few short
        # matches too, so just require it not to blow up
        self.assertLess(len(blob), len(data) * 1.01)
        self.assertEqual(B.decode(blob, len(data)), data)

    def test_repetitive_data_compresses_hard(self):
        data = b"StarCraft 64 " * 4000
        blob = B.encode(data, B.LEVEL_OPTIMAL)
        self.assertEqual(B.decode(blob, len(data)), data)
        self.assertLess(len(blob), len(data) // 100)

    def test_single_byte_repeat_becomes_one_long_run(self):
        data = b"\xa5" * 60000
        blob = B.encode(data, B.LEVEL_OPTIMAL)
        self.assertEqual(B.decode(blob, len(data)), data)
        # one literal plus a handful of bytes of overlapping back-reference
        self.assertLess(len(blob), 24)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@unittest.skipIf(_rom_path() is None,
                 "no ROM: set BOLT_ROM or pass --rom PATH")
class TestAgainstRom(unittest.TestCase):
    """Round-trip real decoded BOLT payloads.

    Nothing here writes game data anywhere; entries are decoded in memory,
    re-encoded, and compared.
    """

    def test_archive_parses(self):
        archive = _archive()
        self.assertIsNotNone(archive, "ROM present but no BOLT archive found")
        entries = list(archive.entries())
        self.assertGreater(len(entries), 100)

    def test_roundtrip_real_entries(self):
        archive = _archive()
        self.assertIsNotNone(archive)
        n = int(os.environ.get("BOLT_ROM_ENTRIES", "60"))
        for level in LEVELS:
            for entry in _sample_entries(n, seed=11):
                plain = archive.read(entry)
                with self.subTest(level=level, entry=entry.path,
                                  size=len(plain)):
                    blob = B.encode(plain, level=level)
                    self.assertEqual(B.decode(blob, len(plain)), plain)

    def test_roundtrip_stored_entries(self):
        """Entries the ROM chose not to compress are the awkward inputs."""
        archive = _archive()
        self.assertIsNotNone(archive)
        stored = [e for e in archive.entries() if e.stored][:40]
        if not stored:
            self.skipTest("this archive has no uncompressed entries")
        for entry in stored:
            plain = archive.read(entry)
            for level in LEVELS:
                with self.subTest(level=level, entry=entry.path):
                    self.assertEqual(
                        B.decode(B.encode(plain, level=level), len(plain)),
                        plain)


@unittest.skipIf(_rom_path() is None,
                 "no ROM: set BOLT_ROM or pass --rom PATH")
@unittest.skipIf(_sc64_decoder() is None,
                 "no reference decoder: set SC64_MAPS to the sc64-maps repo")
class TestDifferentialDecode(unittest.TestCase):
    """Prove this decoder against sc64-maps' proven one before trusting it.

    Round-trip tests only show that the encoder and decoder agree with each
    other.  This test decodes the ROM's *own* compressed bytes -- produced by
    Mass Media's encoder, not ours -- with both implementations and requires
    them to be identical.
    """

    def test_matches_reference_decoder_on_many_entries(self):
        archive = _archive()
        ref_cls = _sc64_decoder()
        self.assertIsNotNone(archive)
        ref = ref_cls(archive.rom)
        self.assertEqual(ref.base, archive.base,
                         "the two readers disagree about the archive offset")

        entries = [e for e in archive.entries() if not e.stored]
        target = int(os.environ.get("BOLT_DIFF_ENTRIES", "250"))
        self.assertGreaterEqual(
            len(entries), 200,
            "need at least 200 compressed entries for the differential test")
        checked = 0
        for entry in entries[:target]:
            mine = B.decode(archive.rom, entry.size,
                            start=archive.base + entry.offset)
            theirs = ref._decompress(entry.offset, entry.size, entry.size)
            self.assertEqual(len(mine), entry.size, entry.path)
            self.assertEqual(mine, theirs, f"divergence in {entry.path}")
            checked += 1
        self.assertGreaterEqual(checked, 200)

    def test_every_entry_decodes_to_its_declared_length(self):
        """The whole archive, not a sample: a strong check on the decoder."""
        if not os.environ.get("BOLT_FULL"):
            self.skipTest("set BOLT_FULL=1 to decode the entire archive")
        archive = _archive()
        for entry in archive.entries():
            if entry.stored:
                continue
            out = B.decode(archive.rom, entry.size,
                           start=archive.base + entry.offset)
            self.assertEqual(len(out), entry.size, entry.path)


if __name__ == "__main__":
    unittest.main(argv=[a for a in sys.argv if a != "--rom"
                        and not os.path.isfile(a)])
