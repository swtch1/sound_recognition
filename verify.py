"""Verify one clip against the enrolled sentinels -- the button-1 (unlock) path.

    python verify.py assets/positives/pos_01.wav        # a match -> exit 0
    python verify.py assets/negatives/neg_01.wav        # no match -> exit 1
    python verify.py some_clip.wav 22                    # override the threshold

Takes a single WAV (mono 16 kHz, the format record.py writes), scores it against
every sentinel in assets/sentinels/, and reports whether it's within threshold.
Exit code is 0 on a match, 1 on no match -- so a caller (a shell script, or GPIO
glue on the Pi) can drive the lock off the return code without parsing output.
"""

import sys

from matcher import features_for, load_sentinels, match, THRESHOLD


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python verify.py <clip.wav> [threshold]")
    clip = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else THRESHOLD

    sentinels = load_sentinels()
    if not sentinels:
        raise SystemExit("No sentinels enrolled -- record one to assets/sentinels/ first.")

    is_match, dist = match(features_for(clip), sentinels, threshold)
    verdict = "MATCH -- unlock" if is_match else "no match -- stay locked"
    print(f"{clip}")
    print(f"  nearest distance {dist:.2f}   threshold {threshold:.1f}   -> {verdict}")
    return 0 if is_match else 1


if __name__ == "__main__":
    sys.exit(main())
