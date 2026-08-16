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


# The cost of a reference depends on the distance only through how many bits
# the offset accumulator needs, so a cache keyed on that width plus the length
# has a very high hit rate -- there are only a couple of dozen distinct widths
# in any realistic window.
_COST_CACHE: dict[tuple[int, int], int | None] = {}
_MISS = object()


def _cost_by_off_bits(bits_off: int, length: int) -> int | None:
    key = (bits_off, length)
    hit = _COST_CACHE.get(key, _MISS)
    if hit is not _MISS:
        return hit

    result = None
    for k in range(0, _MAX_EXT + 1):
        rest = length - k - 2
        if rest < 0:
            break
        bits_run = (rest >> BITS_REF_LEN).bit_length()
        for n_dual in range(0, k + 1):
            need_off = max(0, bits_off - BITS_DUAL * n_dual)
            need_run = max(0, bits_run - BITS_DUAL * n_dual)
            if (-(-need_off // BITS_OFF_EXT)
                    + -(-need_run // BITS_RUN_EXT) + n_dual) <= k:
                result = k + 1
                break
        if result is not None:
            break

    _COST_CACHE[key] = result
    return result


def reference_cost(dist: int, length: int) -> int | None:
    """Encoded size in bytes of a back-reference, or None if unrepresentable."""
    if dist < 1 or length < MIN_MATCH:
        return None
    return _cost_by_off_bits(((dist - 1) >> BITS_REF_DIST).bit_length(),
                             length)


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
# Match finding
# ---------------------------------------------------------------------------

_HASH_BITS = 16
_HASH_SIZE = 1 << _HASH_BITS
_HASH_MASK = _HASH_SIZE - 1

# Distances above this are never searched.  Six offset-extension bytes could
# express far more, but nothing in a cartridge needs it.
DEFAULT_MAX_DIST = 1 << 22

# Longest match the finder will bother measuring.
DEFAULT_MAX_LEN = 1 << 16

# How many chain links to walk before giving up on a position.
DEFAULT_MAX_CHAIN = 64

# Stop walking the chain once a match this long turns up.
DEFAULT_NICE_LEN = 128

# How many lengths past the previous candidate the finder enumerates one by
# one before it jumps straight to the longest match available.  Truncating a
# match is only ever useful to line the next operation up with something
# better, which in practice means shaving a few bytes, and cost(d, l) sits in
# wide plateaus, so enumerating hundreds of lengths costs time and buys
# almost nothing.
_LEN_DETAIL = 48


def _match_len(data, a: int, b: int, limit: int) -> int:
    """Length of the common prefix of data[a:] and data[b:], capped at limit.

    Compares in doubling slices so the byte loop stays in C.  Overlapping
    ranges (b - a < limit) are fine and are exactly how a run is found: the
    decoder copies one byte at a time, so a match may legally extend past its
    own distance.
    """
    if limit <= 0:
        return 0
    n = 0
    step = 8
    while n < limit:
        m = step if step < limit - n else limit - n
        if data[a + n:a + n + m] == data[b + n:b + n + m]:
            n += m
            step += step
        else:
            break
    # refine the failing chunk one byte at a time
    while n < limit and data[a + n] == data[b + n]:
        n += 1
    return n


class _MatchFinder:
    """Hash chains over three-byte prefixes, plus a short scan for pairs.

    Two-byte matches matter in this format -- one costs a single byte at a
    distance of 16 or less -- but they are far too common to hash, so they
    are found by a bounded backward scan instead.
    """

    def __init__(self, data: bytes, max_dist: int, max_chain: int,
                 nice_len: int, max_len: int):
        self.data = data
        self.n = len(data)
        self.max_dist = max_dist
        self.max_chain = max_chain
        self.nice_len = nice_len
        self.max_len = max_len
        self.head = [-1] * _HASH_SIZE
        self.prev = [-1] * (self.n + 1)

    def _hash(self, pos: int) -> int:
        d = self.data
        return ((d[pos] << 10) ^ (d[pos + 1] << 5) ^ d[pos + 2]) & _HASH_MASK

    def insert(self, pos: int) -> None:
        if pos + 2 < self.n:
            h = self._hash(pos)
            self.prev[pos] = self.head[h]
            self.head[h] = pos

    def find(self, pos: int, detail: int = _LEN_DETAIL):
        """Candidate matches at `pos`, as a list of (length, distance).

        For every achievable length up to `detail` the *smallest* distance is
        reported, because a shorter distance is never more expensive.  The
        single longest match is always included even when it exceeds
        `detail`.
        """
        if pos == 0 or pos >= self.n:
            return []
        data = self.data
        limit = self.max_len
        if limit > self.n - pos:
            limit = self.n - pos
        if limit < MIN_MATCH:
            return []

        out = []
        best_len = MIN_MATCH - 1

        # Two-byte matches: only worth anything within 16 bytes, where they
        # cost one byte.  Walk backwards and take the first hit.
        near = pos if pos < MAX_PLAIN_LITERAL else MAX_PLAIN_LITERAL
        first = data[pos]
        second = data[pos + 1] if limit >= 2 else None
        for d in range(1, near + 1):
            p = pos - d
            if data[p] == first and (limit < 2 or data[p + 1] == second):
                ln = _match_len(data, p, pos, limit)
                if ln >= MIN_MATCH:
                    out.append((ln, d))
                    best_len = ln
                break

        if limit >= 3 and pos + 2 < self.n:
            p = self.head[self._hash(pos)]
            chain = self.max_chain
            floor = pos - self.max_dist
            while p >= 0 and p >= floor and chain > 0:
                chain -= 1
                if best_len < limit and data[p + best_len] == data[pos + best_len]:
                    ln = _match_len(data, p, pos, limit)
                    if ln > best_len:
                        out.append((ln, pos - p))
                        best_len = ln
                        if ln >= self.nice_len:
                            break
                p = self.prev[p]

        if not out:
            return []

        # `out` holds strictly increasing lengths, each paired with the
        # smallest distance that reaches it.  Expand into per-length
        # candidates so the parser may choose a shorter match at a cheaper
        # distance, but stop enumerating after `detail` steps per group and
        # jump to the group's full length.
        expanded = []
        prev_len = MIN_MATCH - 1
        for ln, dist in out:
            top = ln if ln < prev_len + detail else prev_len + detail
            for length in range(prev_len + 1, top + 1):
                expanded.append((length, dist))
            if ln > top:
                expanded.append((ln, dist))
            prev_len = ln
        return expanded


# ---------------------------------------------------------------------------
# Encoder -- entry point
# ---------------------------------------------------------------------------

LEVEL_STORE = 0     # literal runs only; the trivially correct baseline
LEVEL_GREEDY = 1    # hash-chain match finding, take the longest match
LEVEL_LAZY = 2      # greedy plus one-byte lookahead
LEVEL_OPTIMAL = 3   # shortest-path parse over the real cost model

DEFAULT_LEVEL = LEVEL_OPTIMAL

ALL_LEVELS = (LEVEL_STORE, LEVEL_GREEDY, LEVEL_LAZY, LEVEL_OPTIMAL)

_LEVEL_TUNING = {
    LEVEL_GREEDY: dict(max_chain=16, nice_len=64),
    LEVEL_LAZY: dict(max_chain=48, nice_len=128),
    LEVEL_OPTIMAL: dict(max_chain=96, nice_len=1024),
}


def encode(data, level: int = DEFAULT_LEVEL, *,
           max_dist: int = DEFAULT_MAX_DIST,
           max_chain: int | None = None,
           nice_len: int | None = None,
           max_len: int = DEFAULT_MAX_LEN) -> bytes:
    """Compress `data` into a BOLT LZSS stream.

    `level` trades encode time for output size:
        0  literal runs only -- correct, never compresses
        1  greedy match finding
        2  lazy match finding (one byte of lookahead)
        3  optimal parse over the real cost model (default)

    The result always satisfies `decode(encode(x)) == x`.
    """
    data = bytes(data)
    if level == LEVEL_STORE:
        return _encode_store(data)
    if level not in _LEVEL_TUNING:
        raise BoltLZSSError(f"unknown compression level {level!r}")
    if not data:
        return b""

    tuning = _LEVEL_TUNING[level]
    finder = _MatchFinder(
        data,
        max_dist=max_dist,
        max_chain=tuning["max_chain"] if max_chain is None else max_chain,
        nice_len=tuning["nice_len"] if nice_len is None else nice_len,
        max_len=max_len,
    )
    if level == LEVEL_OPTIMAL:
        return _encode_optimal(data, finder)
    return _encode_greedy(data, finder, lazy=(level == LEVEL_LAZY))


def _flush_literals(out: bytearray, data: bytes, start: int, end: int) -> None:
    """Emit data[start:end] as literal runs.

    One long run always beats several short ones -- the run length costs a
    control byte plus five bits of extension per five bits of magnitude,
    while splitting costs a whole control byte each time -- so this emits a
    single run whenever it can.
    """
    if end <= start:
        return
    _emit_literal(out, data, start, end - start)


def _encode_greedy(data: bytes, finder: _MatchFinder, lazy: bool) -> bytes:
    """Take the best match at each position; optionally look one byte ahead.

    "Best" is the largest saving, not the longest match: a nearer match can
    be cheaper than a longer far one, because distance is paid for in whole
    extension bytes.
    """
    n = len(data)
    out = bytearray()
    lit_start = 0
    pos = 0
    inserted = 0        # every position below this is already in the chains

    def advance(upto):
        """Insert hash entries up to (not including) `upto`, exactly once.

        Inserting a position twice would make it its own predecessor and the
        chain walk would never terminate, so the watermark is not optional.
        """
        nonlocal inserted
        while inserted < upto:
            finder.insert(inserted)
            inserted += 1

    def best_at(p):
        """(gain, length, dist) for the most profitable match at p."""
        best = (0, 0, 0)
        for length, dist in finder.find(p, detail=16):
            cost = reference_cost(dist, length)
            if cost is None:
                continue
            gain = length - cost
            if gain > best[0] or (gain == best[0] and length > best[1]):
                best = (gain, length, dist)
        return best

    while pos < n:
        gain, length, dist = best_at(pos)
        # Require two bytes of saving: taking a match also terminates the
        # literal run, and the run that resumes afterwards needs a fresh
        # control byte.
        if gain >= 2:
            if lazy and pos + 1 < n:
                advance(pos + 1)
                if best_at(pos + 1)[0] > gain:
                    pos += 1
                    continue
            _flush_literals(out, data, lit_start, pos)
            _emit_reference(out, dist, length)
            pos += length
            advance(min(pos, n))
            lit_start = pos
        else:
            advance(pos + 1)
            pos += 1

    _flush_literals(out, data, lit_start, n)
    return bytes(out)


# Literal-run cost tiers: (longest run at this tier, extension bytes used).
# A run of L bytes costs L + 1 + t.  The bounds come straight from the field
# widths: ((ext_run << 4) | 0x0F) + 1 with ext_run holding 5*t bits.
def _literal_tiers(n: int):
    tiers = []
    for t in range(0, 5):
        span = (((1 << (BITS_RUN_EXT * t)) - 1) << BITS_LIT_LEN | 0x0F) + 1
        tiers.append((span, t))
        if span >= n:
            break
    return tiers


def _encode_optimal(data: bytes, finder: _MatchFinder) -> bytes:
    """Shortest-path parse over the format's real cost model.

    Nodes are byte positions, edges are operations, edge weights are exactly
    what the operation costs on the wire.  Two things make this less routine
    than the usual LZ optimal parse:

      * A literal run's cost is not the sum of its bytes -- it has a control
        byte plus a step function of extension bytes -- so literal edges are
        relaxed with a sliding-window minimum per cost tier rather than one
        edge per byte.

      * A back-reference's cost depends on the length *and* the distance
        together, through op_count, so `reference_cost` has to be consulted
        per candidate rather than assumed monotone.
    """
    from collections import deque

    n = len(data)
    inf = float("inf")
    best = [inf] * (n + 1)
    best[0] = 0
    from_pos = [0] * (n + 1)
    ref_dist = [0] * (n + 1)       # 0 marks a literal-run edge

    tiers = _literal_tiers(n)
    windows = [deque() for _ in tiers]

    for i in range(0, n + 1):
        if i > 0:
            # ---- literal-run edges landing on i -------------------------
            for (span, t), dq in zip(tiers, windows):
                lo = i - span
                while dq and dq[0][0] < lo:
                    dq.popleft()
                if dq:
                    j, v = dq[0]
                    cand = v + i + 1 + t
                    if cand < best[i]:
                        best[i] = cand
                        from_pos[i] = j
                        ref_dist[i] = 0

        cur = best[i]
        if cur == inf:
            continue
        if i < n:
            # best[i] is final now: every edge into i came from a smaller
            # index and has already been relaxed.
            key = cur - i
            for dq in windows:
                while dq and dq[-1][1] >= key:
                    dq.pop()
                dq.append((i, key))

            # ---- back-reference edges leaving i -------------------------
            for length, dist in finder.find(i):
                cost = reference_cost(dist, length)
                if cost is None:
                    continue
                j = i + length
                cand = cur + cost
                if cand < best[j]:
                    best[j] = cand
                    from_pos[j] = i
                    ref_dist[j] = dist
            finder.insert(i)

    if best[n] == inf:                       # unreachable in practice
        raise BoltLZSSError("optimal parse failed to reach the end of input")

    # ---- walk the path back and emit -----------------------------------
    path = []
    j = n
    while j > 0:
        i = from_pos[j]
        path.append((i, j, ref_dist[j]))
        j = i
    path.reverse()

    out = bytearray()
    for i, j, dist in path:
        if dist:
            _emit_reference(out, dist, j - i)
        else:
            _emit_literal(out, data, i, j - i)
    return bytes(out)


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
