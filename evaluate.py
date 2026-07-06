"""POC harness: enroll the sentinels, score every clip, print a table.

Layout under assets/:
  sentinels/  enrolled reference(s) -- the known-good sound
  positives/  separate genuine takes that SHOULD match
  negatives/  sounds that should NOT match

Each test clip's score is its DTW distance to the *nearest* enrolled sentinel.
Read the table like this: every negative should score well above every genuine
positive. The gap between the two groups is what a threshold lives in.
"""

import glob
import os

from matcher import dtw_distance, features_for

ROOT = "assets"


def load_dir(name):
    files = sorted(glob.glob(os.path.join(ROOT, name, "*.wav")))
    return [(f, features_for(f)) for f in files]


def score(feat, templates, exclude=None):
    """Nearest DTW distance from `feat` to any template (optionally excluding one)."""
    dists = [dtw_distance(feat, t) for f, t in templates if f != exclude]
    return min(dists) if dists else None


def main():
    sentinels = load_dir("sentinels")
    positives = load_dir("positives")
    negatives = load_dir("negatives")
    if not sentinels:
        raise SystemExit(f"No sentinels enrolled -- put WAVs in {ROOT}/sentinels/")

    rows = []
    for f, feat in positives:
        rows.append((score(feat, sentinels), "POSITIVE", f))
    for f, feat in negatives:
        rows.append((score(feat, sentinels), "negative", f))

    print(f"\nEnrolled {len(sentinels)} sentinel(s), scoring "
          f"{len(positives)} positive + {len(negatives)} negative clip(s).\n")
    print(f"{'score':>8}  {'label':<9}  file")
    print("-" * 48)
    for dist, label, f in sorted(rows, key=lambda r: (r[0] is None, r[0])):
        shown = "  n/a" if dist is None else f"{dist:7.2f}"
        print(f"{shown:>8}  {label:<9}  {f}")

    pos = [d for d, l, _ in rows if l == "POSITIVE" and d is not None]
    neg = [d for d, l, _ in rows if l == "negative" and d is not None]
    print()
    if pos and neg:
        margin = min(neg) - max(pos)
        print(f"worst positive: {max(pos):.2f}   best negative: {min(neg):.2f}"
              f"   margin: {margin:+.2f}")
        print("A threshold anywhere in that margin separates the two groups."
              if margin > 0 else
              "OVERLAP -- no clean threshold; the features aren't separating these.")
    else:
        print("Only self/trivial positives available -- need extra sentinel "
              "takes to measure genuine-positive scores.")


if __name__ == "__main__":
    main()
