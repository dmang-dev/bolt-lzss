#!/usr/bin/env python3
"""Tally what the original encoder actually emitted.

Having a decoder tells you what a stream means.  Walking every stream in an
archive and counting the operations tells you something else: what the
*encoder* on the other side was like -- which opcodes it bothered to use, how
big a window it searched, how long it let a run get.

This is the tool behind the "What the cartridge's own encoder does" section
of the README.

Usage:
    py -3.13 analyse_rom.py --rom PATH

Reads the ROM, writes nothing.  No game data is extracted or reported -- only
counts, sizes and distances.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bolt_archive


def walk_stream(rom, start: int, want: int, ops: Counter,
                lit_runs: Counter, match_len: Counter,
                match_dist: Counter, match_ext: Counter) -> None:
    """Decode one stream for its statistics, discarding the output.

    This is a second, deliberately separate implementation of the decoder's
    dispatch: it tracks output *length* rather than content, so it can walk
    48 MB of archive without materialising it.
    """
    pos = start
    produced = 0
    op_count = ext_offset = ext_run = 0

    while produced < want:
        b = rom[pos]
        pos += 1
        op_count += 1

        if b & 0x80:
            if b & 0x40:
                ops["offset extension (11xxxxxx)"] += 1
                ext_offset = (ext_offset << 6) | (b & 0x3F)
            elif b & 0x20:
                ops["run extension (101xxxxx)"] += 1
                ext_run = (ext_run << 5) | (b & 0x1F)
            elif b & 0x10:
                ops["dual extension (1001xxxx)"] += 1
                ext_run = (ext_run << 2) | (b & 0x03)
                ext_offset = (ext_offset << 2) | ((b >> 2) & 0x03)
            else:
                run = ((ext_run << 4) | (b & 0x0F)) + 1
                ops["literal run"] += 1
                ops["bytes from literal runs"] += run
                lit_runs[run] += 1
                produced += run
                pos += run
                op_count = ext_offset = ext_run = 0
        else:
            dist = ((ext_offset << 4) | (b & 0x0F)) + 1
            run = ((ext_run << 3) | (b >> 4)) + op_count + 1
            ops["back-reference"] += 1
            ops["bytes from back-references"] += run
            ops["length credited by op_count"] += op_count + 1
            match_len[run] += 1
            match_dist[dist] += 1
            match_ext[op_count - 1] += 1
            produced += run
            op_count = ext_offset = ext_run = 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", default=os.environ.get("BOLT_ROM"),
                    help="path to an N64 dump containing a BOLT archive")
    args = ap.parse_args(argv)
    if not args.rom or not os.path.isfile(args.rom):
        ap.error("no ROM: pass --rom PATH or set BOLT_ROM")

    rom = bolt_archive.load_rom(args.rom)
    archive = bolt_archive.BoltArchive(rom)

    ops = Counter()
    lit_runs = Counter()
    match_len = Counter()
    match_dist = Counter()
    match_ext = Counter()
    n = 0
    for entry in archive.entries():
        if entry.stored:
            continue
        walk_stream(rom, archive.base + entry.offset, entry.size,
                    ops, lit_runs, match_len, match_dist, match_ext)
        n += 1

    print(f"archive at {archive.base:#x}, built {archive.build_stamp}")
    print(f"{n} compressed entries\n")

    print("operation counts")
    for key in sorted(ops):
        print(f"  {key:<34} {ops[key]:>14,}")

    matched = ops["bytes from back-references"]
    credited = ops["length credited by op_count"]
    print(f"\n  op_count bias covers {credited:,} of {matched:,} matched "
          f"bytes ({credited / matched:.1%})")
    print("  -- that is output paid for by the length bias rather than by "
          "encoded length bits")

    print("\nliteral runs")
    print(f"  distinct lengths     {len(lit_runs):>10,}")
    print(f"  runs of exactly 1    {lit_runs[1]:>10,}  "
          f"({lit_runs[1] / sum(lit_runs.values()):.1%} of all runs)")
    print(f"  longest run          {max(lit_runs):>10,}")

    print("\nback-references")
    print(f"  shortest             {min(match_len):>10,}")
    print(f"  longest              {max(match_len):>10,}")
    print(f"  nearest distance     {min(match_dist):>10,}")
    print(f"  furthest distance    {max(match_dist):>10,}")
    total = sum(match_dist.values())
    for bound in (16, 1024, 65536, 262144, 1 << 22):
        hits = sum(v for k, v in match_dist.items() if k <= bound)
        print(f"    distance <= {bound:>9,}  {hits:>12,}  "
              f"({hits / total:6.2%})")
    print("  extension bytes per reference:")
    for k in sorted(match_ext):
        print(f"    k = {k}  {match_ext[k]:>12,}")

    print("\nThe format expresses distances to 4,194,304 with three offset "
          "extension bytes.")
    print(f"This encoder never exceeded {max(match_dist):,}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
