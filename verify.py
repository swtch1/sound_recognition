"""Verify an unlock attempt against one or more sentinels -- the button-1 path.

    uv run python verify.py --sentinel good.wav --attempt try.wav
    uv run python verify.py --sentinel a.wav --sentinel b.wav --attempt try.wav

Both --sentinel (repeatable, one per known-good sound) and --attempt are WAV
files (mono 16 kHz, the format record.py writes). Exit code is 0 when the
attempt matches a sentinel within threshold, 1 when it does not -- so a caller
(a shell script, or GPIO glue on the Pi) can drive the lock off the return code
alone, without parsing the printed line.
"""

import argparse
import sys

from matcher import features_for, match, THRESHOLD


def main():
    p = argparse.ArgumentParser(
        description="Verify an unlock attempt against sentinel sound(s).")
    p.add_argument("--sentinel", action="append", required=True, metavar="WAV",
                   help="a known-good sound; pass repeatedly for multiple sentinels")
    p.add_argument("--attempt", required=True, metavar="WAV",
                   help="the unlock attempt to verify")
    p.add_argument("--threshold", type=float, default=THRESHOLD, metavar="D",
                   help=f"DTW distance cutoff (default {THRESHOLD})")
    args = p.parse_args()

    sentinels = [features_for(s) for s in args.sentinel]
    is_match, dist = match(features_for(args.attempt), sentinels, args.threshold)

    verdict = "MATCH -- unlock" if is_match else "no match -- stay locked"
    print(f"attempt {args.attempt}")
    print(f"  nearest of {len(sentinels)} sentinel(s): {dist:.2f}   "
          f"threshold {args.threshold:.1f}   -> {verdict}")
    return 0 if is_match else 1


if __name__ == "__main__":
    sys.exit(main())
