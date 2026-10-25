#!/usr/bin/env python
"""Regenerate graphs from whatever is on disk. Never touches the GPU.

  python plots.py                     all graphs -> out/plots/
  python plots.py --list              show available graph names
  python plots.py train_loss speedup  only these
  python plots.py --stamp             timestamped filenames (keeps history)
"""
from __future__ import annotations
import argparse
import sys

from draftforge.paths import P, ensure_dirs, pin_caches
from draftforge.util import setup_logging, now_ts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="plots.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="graph names (default: all)")
    ap.add_argument("--list", action="store_true", help="list graph names and exit")
    ap.add_argument("--stamp", action="store_true",
                    help="write timestamped filenames instead of overwriting")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    ensure_dirs()
    pin_caches()
    setup_logging(P.logs / ("plots_%s.log" % now_ts()), args.verbose)

    from draftforge import plots
    if args.list:
        print("available graphs:")
        for n in plots.PLOTS:
            print("  " + n)
        return 0

    files = plots.make(args.names or None, stamp=args.stamp)
    for f in files:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
