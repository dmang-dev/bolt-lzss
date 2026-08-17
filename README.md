# bolt-lzss

An encoder **and** decoder for the LZSS variant used inside Mass Media's BOLT
archives.

The decoder half has existed since Adam Heinermann reverse engineered the
format for [BOLTextract](https://github.com/heinermann/BOLTextract). The
encoder half, as far as I can find, has not existed anywhere. BOLTextract is
decode-only, the format is Mass Media's own rather than a licensed library,
and nothing published compresses into it. This repository is an attempt at
the missing direction.

The codec is a single file, `bolt_lzss.py` — standard library only, no
dependencies, nothing to install. Drop it in and import it.

```python
import bolt_lzss

blob  = bolt_lzss.encode(payload)          # bytes -> compressed bytes
plain = bolt_lzss.decode(blob, len(payload))
assert plain == payload
```

---

## What BOLT is

BOLT is the archive container Mass Media Interactive Entertainment (and
Philips P.O.V.) embedded in their games from the CD-i era through to the Xbox
one. It holds a small directory tree of unnamed, typed, individually
compressed files, and it is usually welded directly into the executable or
the cartridge image rather than shipped as a separate file.

BOLTextract identifies five compression algorithms across the family. This
project implements the one it calls `n64`/`gba`, used in titles released
between roughly 1999 and 2003 — and, per BOLTextract's notes, the same
algorithm that the `xbox`/`ps2` archives from 2004 onward use, those differing
only in their container structures. So the codec here should apply unchanged
to the later archives once you have parsed their headers.

Games known to use BOLT archives include Bassmasters 2000, Ms. Pac-Man Maze
Madness, Power Rangers Lightspeed Rescue, Namco Museum, StarCraft 64,
Blackthorne, Rock n' Roll Racing, The Lost Vikings, Pac-Man Collection and
Shrek Super Party. This project was developed and measured against the
StarCraft 64 cartridge, whose BOLT archive holds 2111 entries.

No game data is included here, and none can be committed — `.gitignore`
blocks every ROM container extension outright.

---

## The bitstream

There is no header, no magic, no end marker and no stored length. A stream is
a bare sequence of operations; the decoder stops when it has produced the
number of bytes the *container* said to expect.

Decoding keeps three accumulators, all reset to zero after every operation
that emits output:

| accumulator | meaning |
|---|---|
| `op_count` | control bytes consumed since the last emit |
| `ext_offset` | pending high bits of a back-reference distance |
| `ext_run` | pending high bits of a run length |

Control bytes are read one at a time and dispatched on their high bits:

| bits | range | operation |
|---|---|---|
| `0xxxxxxx` | `0x00`–`0x7F` | **back-reference** — `distance = ((ext_offset << 4) \| (b & 0x0F)) + 1`, `length = ((ext_run << 3) \| (b >> 4)) + op_count + 1` |
| `1000xxxx` | `0x80`–`0x8F` | **literal run** of `((ext_run << 4) \| (b & 0x0F)) + 1` bytes, taken from the stream immediately after this byte |
| `1001xxxx` | `0x90`–`0x9F` | **extend both**: `ext_run = (ext_run << 2) \| (b & 3)`, `ext_offset = (ext_offset << 2) \| ((b >> 2) & 3)` |
| `101xxxxx` | `0xA0`–`0xBF` | **extend run** by five bits: `ext_run = (ext_run << 5) \| (b & 0x1F)` |
| `11xxxxxx` | `0xC0`–`0xFF` | **extend offset** by six bits: `ext_offset = (ext_offset << 6) \| (b & 0x3F)` |

Back-reference copies are performed **one byte at a time**, so a run may
overlap itself and repeat — that is how the format spells RLE. A distance of
1 with a long length repeats the last byte.

Because the accumulators shift left, extension bytes carry the *more*
significant bits and must be emitted before the bits they precede.

Since there is no end marker, nothing in the format stops a final operation
from overshooting the declared length — a decoder would simply produce more
bytes than the container asked for. In practice that never happens: across
all four StarCraft 64 releases, all 4,503 compressed entries land exactly on
their declared size, so this decoder treats an overshoot as an error by
default. The original encoder was evidently length-aware at the tail.

---

## The op_count subtlety

This is the part that is easy to miss reading a decoder, and it is what makes
the format unusual.

`op_count` counts every control byte since the last emit — **including the
back-reference byte itself** — and it is added to the back-reference length.
The extension bytes you spend widening a distance are therefore not pure
overhead; they are also credited to the run. If a back-reference is preceded
by `k` extension bytes:

```
length = ((ext_run << 3) | (b >> 4)) + k + 2
```

Three consequences follow, all of which an encoder must model and a decoder
never has to think about.

**1. Length and distance are not independent.** You cannot pick the encoded
run field without knowing how many bytes the distance will cost, and you
cannot pick the distance encoding without knowing the run, because run
extension bytes also inflate `k`. It is a fixed point. This encoder resolves
it by searching `k` upward from zero and taking the first value that admits a
legal byte layout, which is also the cheapest, since a reference costs exactly
`k + 1` bytes.

**2. Far matches have a minimum length.** A distance above 16 needs at least
one extension byte, so it cannot express a length below 3. Above 1024 it needs
two, and the floor becomes 4; above 65536, 5. Short matches at long range are
not merely unprofitable, they are *unrepresentable*, and an encoder that does
not know this will emit streams that decode to the wrong thing.

**3. A back-reference can never make the stream bigger.** A reference costs
`k + 1` bytes and its shortest expressible length is `k + 2`. So the worst any
legal match can do is save one byte against spelling the same span out as
literals. Every control byte spent on a match buys back exactly one byte of
output. Across the whole StarCraft 64 archive this bias accounts for 10.5 MB
of the 44.0 MB that back-references produce — roughly a quarter of all matched
output is paid for by the bias rather than by encoded length bits.

I have not seen this trick in another LZSS derivative. It is a neat way to
stop the extension-byte machinery from eating the gains it enables.

---

## Emission order is a compatibility constraint

Nothing in the bitstream description above says what order extension bytes must
come in. Any order that reconstructs the same accumulators decodes to the same
output, and a decoder written from that description accepts all of them.

The cartridge's decoder does not. It requires one total order —

```
dual (1001xxxx)  ->  offset (11xxxxxx)  ->  run (101xxxxx)  ->  terminal
```

— and it will not accept more than **two** offset-extension bytes before a
single back-reference.

This is not inferable from the format; it came from profiling every compressed
entry on the StarCraft 64 cartridge and then confirming it on the real engine:

| pattern before a back-reference | cartridge | this encoder before 0.2.0 |
|---|---|---|
| `dual, offset` | 12.08% | 0% |
| `offset, dual` | 0% | 3.01% |
| `offset, offset, offset` | never | present |

The arithmetic corroborates the cap. The widest offset accumulator anywhere on
the cartridge is 15,481 — exactly the 14 bits that one dual byte plus two
offset bytes provide. Three offset bytes would express 18, and nothing on the
cartridge ever does.

**Versions before 0.2.0 emitted offset bytes first**, and so produced streams
that round-trip perfectly through this module's own decoder and hang StarCraft
64 on its loading screen. If you are building a ROM, you need 0.2.0 or later.

There is a general lesson in it. For a codec whose real consumer is someone
else's decoder, `decode(encode(x)) == x` measured against *your own* decoder is
not a correctness test — it only proves the two agree with each other. Both can
be wrong together, and here they were, for every stream the encoder had ever
produced. The test that caught it compares against the original encoder's
output rather than against this one's decoder.

---

## What the cartridge's own encoder does

Having an encoder makes it possible to ask what the original one was like.
Decoding all 1125 compressed entries of the USA cartridge and tallying the
operations gives:

| | count |
|---|---|
| back-references | 3,043,973 |
| literal runs | 966,532 |
| offset-extension bytes (`11xxxxxx`) | 3,301,470 |
| run-extension bytes (`101xxxxx`) | 441,339 |
| dual-extension bytes (`1001xxxx`) | 705,818 |
| bytes produced by back-references | 43,962,757 |
| bytes produced by literal runs | 4,394,934 |

Some things that fall out of it:

* **All four operations are live.** The dual-extension byte in particular is
  not vestigial — it is used 705,818 times, and an encoder that ignored it
  would be leaving bytes on the table whenever both accumulators need
  widening at once.

* **Matches carry almost everything.** Only 9% of the output comes from
  literal runs, and a third of those runs are a single byte long. This is a
  format that expects to be fed data full of structure.

* **The original encoder used a window of about 256 KiB.** No back-reference
  anywhere in the archive reaches further than 262,128 bytes, which is
  256 KiB minus 16, and the distance histogram stops dead there. The format
  itself expresses distances up to 4 MiB with three extension bytes, and 70
  entries — half the archive by volume — are larger than 256 KiB, so this is
  a limit in their encoder rather than in the format. Why the ceiling sits
  exactly 16 bytes below the round number is unknown.

  This project searches up to 4 MiB by default, which is part of why the gap
  in its favour widens on large entries.

* **They never emitted a literal run longer than 1100 bytes**, though the
  format reaches 512 with one extension byte and 16384 with two. Also
  unexplained.

---

## Usage

### Install

Nothing needs installing — `bolt_lzss.py` is one stdlib-only file, and copying
it into your project is a perfectly good way to use it. If you would rather
have it on the path:

```bash
pip install git+https://github.com/dmang-dev/bolt-lzss
```

which also puts a `bolt-lzss` command on your `$PATH`.

Python 3.11 or newer, tested on 3.11 through 3.14 across Linux and Windows.
The codec uses nothing that would trouble an older interpreter — the floor is
just where security support is, since 3.9 went end of life in October 2025 and
3.10 follows in October 2026.

### From the shell

```bash
bolt-lzss encode payload.bin              # -> payload.bin.bolt
bolt-lzss encode payload.bin out.bolt -l 1
bolt-lzss decode out.bolt payload.bin     # the size argument is optional
bolt-lzss decode out.bolt payload.bin -s 65536
```

`encode` verifies the round trip before it writes anything, so it will not
hand you a file it cannot read back.

### As a library

```python
import bolt_lzss

# Compression levels: 0 store, 1 greedy, 2 lazy, 3 optimal parse (default)
blob = bolt_lzss.encode(data)
blob = bolt_lzss.encode(data, level=bolt_lzss.LEVEL_GREEDY)

# The length comes from the container, not the stream.
data = bolt_lzss.decode(blob, expected_size)

# Or decode until the input runs out, when you have nothing else to go on.
data = bolt_lzss.decode(blob)

# Decode in place out of a larger buffer, and measure the stream's own size.
data = bolt_lzss.decode(rom, entry.size, start=archive.base + entry.offset)
n    = bolt_lzss.decoded_length(rom, entry.size, start=...)
```

`decode(encode(x)) == x` holds for every input, at every level.

Three support files come along for the ride:

* `bolt_archive.py` — a small read-only reader for the BOLT container itself
  (header, directory walk, entry flags), enough to point the codec at real
  data.
* `bench_rom.py` — measures this encoder against a cartridge's own compressed
  sizes.
* `analyse_rom.py` — profiles the *original* encoder by tallying the
  operations it emitted. Every number in the section above comes from it.

```
python bench_rom.py   --rom /path/to/rom.n64 --all-levels
python analyse_rom.py --rom /path/to/rom.n64
python -m unittest                      # ROM tests skip if no ROM is set
BOLT_ROM=/path/to/rom.n64 python -m unittest
```

---

## How well does it compress?

Ground truth is the cartridge. Every compressed BOLT entry is a stream that
Mass Media's own encoder produced, and we can recover its exact input, so the
comparison is on byte-identical data.

Compressed sizes are measured by decoding each stream and reporting how many
input bytes it consumed. Differencing consecutive entry data offsets is the
easier route but folds in inter-entry alignment padding, which would flatter
this project by about 0.1%.

### The whole archive

Every compressed entry in the StarCraft 64 (USA) cartridge — 1125 streams,
48,357,691 bytes of payload. Mass Media's encoder packs that into
**12,854,066 bytes**, a ratio of 0.2658.

| level | output | ratio | vs ROM | smaller / equal / larger | speed |
|---|---|---|---|---|---|
| 0 store | 48,547,614 | 1.0039 | **+277.7%** | 0 / 0 / 1125 | 262 MiB/s |
| 1 greedy | 13,927,242 | 0.2880 | **+8.3%** | 24 / 110 / 991 | 499 KiB/s |
| 2 lazy | 13,646,391 | 0.2822 | **+6.2%** | 64 / 114 / 947 | 288 KiB/s |
| 3 optimal | 12,337,765 | 0.2551 | **−4.0%** | 990 / 129 / 6 | 17 KiB/s |

So the optimal parse comes in 4.0% *smaller* than the encoder Mass Media
shipped, on identical input, and is smaller or equal on 1119 of 1125 entries.

That was not the goal — getting within a sane factor was — and it should be
read with the obvious caveat attached: their encoder had to run on 1999
hardware inside a build pipeline, and mine gets to spend 46 minutes of a
modern desktop core on the same 46 MiB. The interesting part is not that a
slow encoder beats a fast one, it is that the format had this much room left
in it and that the room is reachable.

### Where the win comes from, and where it does not

| sample | optimal vs ROM |
|---|---|
| 120 entries under 20 KB | −2.0% (smaller on 98, equal on 22, larger on 0) |
| 12 entries over 40 KB | −4.3% (smaller on 12 of 12) |
| all 1125 entries | −4.0% |

The advantage grows with entry size, which fits the 256 KiB window the
original encoder appears to have used: the bigger the entry, the more of it
their match finder could not see.

The six entries where this encoder loses are all degenerate — mostly-constant
buffers that compress to a couple of dozen bytes either way. The worst is a
307,216-byte entry that is essentially one long fill: the ROM spells it as a
single 307,200-byte back-reference in 18 bytes, while this encoder caps match
length at 65,536 and needs 33. Fifteen bytes, and the cap is what keeps the
match finder from going quadratic on constant data, so it stays.

<!--BENCH-->

Decoding runs at roughly 8.4 MiB/s, dominated by the byte-at-a-time
overlapping copy the format requires. Encoding is pure Python and slower, as
the table shows: the optimal parse took 46 minutes of one desktop core to
work through the archive. Optimal throughput varies a lot with entry size —
63 KiB/s on entries of a few kilobytes, 15 KiB/s on entries of hundreds —
because deeper chains and longer matches both cost more per byte.

### Where the remaining difference lives

The parse is a shortest path over a DAG whose edge weights are exact, so
given a complete candidate set it would be genuinely optimal. It is optimal
only with respect to the candidates the match finder offers it, and that is
where the remaining gap lives.

Measured on 120 cartridge entries, varying the two knobs that control the
candidate set:

| setting | output | vs ROM |
|---|---|---|
| chain depth 96 (default) | 100,820 | 0.9796x |
| chain depth 256 | 100,714 | 0.9785x |
| chain depth 1024 | 100,654 | 0.9780x |

Deeper chains find a nearer distance for the same length and are worth about
0.1% for four times the work, which is why the default sits where it does.

The other knob turns out not to matter at all. The finder enumerates match
lengths individually only for a window past each candidate before jumping
straight to the longest match available, on the theory that truncating a
match to some intermediate length is occasionally the right move. Setting
that window to 16, 48, 128 or 256 produces **byte-identical output** on all
120 entries. Truncation is apparently never worth it here, and the reason is
the same op_count bias as everywhere else: since no match can lose, and the
cost function sits in wide plateaus, a longer match is never more expensive
than a shorter one at the same distance. The knob stays, at a small value,
because it costs nothing.

---

## Correctness

`python -m unittest` runs the suite. Everything needing game data skips
cleanly when there is none, so a bare checkout passes on a machine that has
never seen a cartridge.

* **Round-trip** across random data at many sizes and entropies, empty input,
  one byte, all-identical, highly repetitive, long runs, and long-distance
  repeats — at all four levels.
* **Field-boundary cases** at every size where a format field rolls over
  (16, 512, 1024, 16384 …).
* **Exhaustive reference planning** over every representable
  (distance, length) pair in a small space, plus a randomised sweep of a large
  one, checking that the emitted bytes decode back to exactly the run
  requested and that the planner's cost prediction matches the bytes emitted.
* **Round-trip on real decoded BOLT entries**, pulled from a ROM at runtime
  via `BOLT_ROM`.
* **A differential decode test**: this decoder and sc64-maps' independently
  written, previously proven decoder are both pointed at the ROM's *own*
  compressed bytes — produced by Mass Media's encoder, not ours — and must
  agree. Round-trip tests only prove that an encoder and its decoder share a
  delusion; this one does not.
* **The encoder cross-validated the other way**: streams this encoder
  produces are handed to sc64-maps' decoder, which must return the original.
  This is the check that matters most, because it is the direction any real
  consumer of these streams takes, and because a decoder and an encoder
  written by the same person from the same notes can share a misreading that
  a round-trip will never catch.

The differential test passes on all 1125 compressed entries of the USA
cartridge, and every entry in its 2111-entry archive decodes to its declared
length. Run by hand across all four StarCraft 64 releases — USA, Australia,
USA Beta and the German prototype — the two decoders agree byte for byte on
**4,503 compressed entries totalling 193,153,156 bytes**, with no mismatches.

Round-trip pass rate is 100%: every synthetic case, every real BOLT entry
tested, at every level, `decode(encode(x)) == x`.

---

## Credit

The format was reverse engineered by **Adam Heinermann** in
[BOLTextract](https://github.com/heinermann/BOLTextract), whose `n64.cpp`
carries the comment "entirely guessed". Everything this project knows about
the bitstream traces back to that work. The container documentation in
`bolt_archive.py` comes from the same source.

The decoder here was written from a written specification rather than ported
line by line, so that the project stands alone and so that the differential
test is a meaningful check rather than a tautology — but the *semantics* are
Heinermann's discovery, not mine. The encoder is original.

The differential test compares against the decoder in
[sc64-maps](https://github.com/dmang-dev/sc64-maps), which is itself a port of
BOLTextract's.

---

## Licence

**GPL-3.0-or-later.** See [LICENSE](LICENSE).

The reasoning, since it is worth stating rather than assuming:

A file format is a set of facts, and facts are not copyrightable. An encoder
is not a derivative work of a decoder in any interesting sense — it is the
other direction, and the parsing, match finding and cost modelling here have
no counterpart in BOLTextract. On a narrow reading this project could carry
any licence at all.

But the specification the decoder was written from was itself derived by
reading BOLTextract's GPL-3.0 source. Reimplementing from a specification that
someone else extracted from GPL code is close enough to the line that arguing
about which side of it you are on is a waste of everyone's time. GPL-3.0-or-later
costs nothing here, keeps the work in the same ecosystem as the reverse
engineering that made it possible, and removes the question. If you need this
under other terms, the person to ask is Adam Heinermann, not me.

No game data, no assets and no cartridge content are in this repository, and
nothing in it is licensed to you by Blizzard or Nintendo.
