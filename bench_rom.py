#!/usr/bin/env python3
"""Measure this encoder against the ROM's own compressed sizes.

The cartridge is ground truth: every compressed BOLT entry is a stream that
Mass Media's own encoder produced from data we can recover exactly.  So for
each entry we can decode it, re-encode the result, and compare sizes on
identical input.

Compressed sizes are measured by decoding and reporting how many input bytes
the stream consumed.  Differencing consecutive data offsets is the usual
shortcut but folds in inter-entry alignment padding, which flatters us.

Usage:
    py -3.13 bench_rom.py --rom PATH [--level N] [--sample N] [--max-size N]

No game data is read out of, or written into, this repository.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bolt_archive
import bolt_lzss


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n} B"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", default=os.environ.get("BOLT_ROM"),
                    help="path to an N64 cartridge dump containing a BOLT archive")
    ap.add_argument("--level", type=int, default=bolt_lzss.LEVEL_OPTIMAL,
                    help="0 store, 1 greedy, 2 lazy, 3 optimal (default 3)")
    ap.add_argument("--sample", type=int, default=200,
                    help="how many compressed entries to test (0 = all)")
    ap.add_argument("--max-size", type=int, default=0,
                    help="skip entries larger than this many bytes (0 = no cap)")
    ap.add_argument("--min-size", type=int, default=0,
                    help="skip entries smaller than this many bytes")
    ap.add_argument("--seed", type=int, default=20240607)
    ap.add_argument("--all-levels", action="store_true",
                    help="run every level over the same sample")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not args.rom or not os.path.isfile(args.rom):
        ap.error("no ROM: pass --rom PATH or set BOLT_ROM")

    rom = bolt_archive.load_rom(args.rom)
    archive = bolt_archive.BoltArchive(rom)
    print(f"archive at {archive.base:#x}, built {archive.build_stamp}")

    entries = [e for e in archive.entries() if not e.stored]
    if args.max_size:
        entries = [e for e in entries if e.size <= args.max_size]
    if args.min_size:
        entries = [e for e in entries if e.size >= args.min_size]
    if args.sample and len(entries) > args.sample:
        entries = random.Random(args.seed).sample(entries, args.sample)
    entries.sort(key=lambda e: e.path)
    print(f"{len(entries)} compressed entries selected")

    payloads = []
    for e in entries:
        plain = archive.read(e)
        rom_size = archive.compressed_length(e)
        payloads.append((e, plain, rom_size))

    levels = (bolt_lzss.ALL_LEVELS if args.all_levels else (args.level,))
    names = {0: "store", 1: "greedy", 2: "lazy", 3: "optimal"}

    for level in levels:
        raw_total = rom_total = ours_total = 0
        wins = ties = 0
        worst = None
        t0 = time.perf_counter()
        for entry, plain, rom_size in payloads:
            blob = bolt_lzss.encode(plain, level=level)
            back = bolt_lzss.decode(blob, len(plain))
            if back != plain:
                print(f"ROUND-TRIP FAILURE in {entry.path}")
                return 1
            raw_total += len(plain)
            rom_total += rom_size
            ours_total += len(blob)
            if len(blob) < rom_size:
                wins += 1
            elif len(blob) == rom_size:
                ties += 1
            ratio = len(blob) / rom_size if rom_size else 1.0
            if worst is None or ratio > worst[0]:
                worst = (ratio, entry.path, len(plain), rom_size, len(blob))
            if args.verbose:
                print(f"  {entry.path:>10}  raw {len(plain):>8}  "
                      f"rom {rom_size:>8}  ours {len(blob):>8}  "
                      f"{len(blob) / rom_size:6.3f}x")
        dt = time.perf_counter() - t0

        print()
        print(f"level {level} ({names.get(level, '?')})")
        print(f"  raw            {raw_total:>12,}  ({human(raw_total)})")
        print(f"  ROM's encoder  {rom_total:>12,}  "
              f"ratio {rom_total / raw_total:.4f}")
        print(f"  this encoder   {ours_total:>12,}  "
              f"ratio {ours_total / raw_total:.4f}")
        print(f"  ours / ROM     {ours_total / rom_total:>12.4f}"
              f"   ({(ours_total / rom_total - 1) * 100:+.1f}%)")
        print(f"  smaller than the ROM on {wins}/{len(payloads)} entries, "
              f"equal on {ties}")
        if worst:
            print(f"  worst entry    {worst[1]} raw {worst[2]} "
                  f"rom {worst[3]} ours {worst[4]} ({worst[0]:.3f}x)")
        print(f"  {dt:.1f}s  ({raw_total / dt / 1024:,.0f} KiB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
