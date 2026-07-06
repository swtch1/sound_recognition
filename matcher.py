"""Core sound-matching pipeline: load -> features -> DTW distance.

The recognition idea, in three steps:
  1. Reduce each clip to MFCCs -- compact features that model how a human
     hears, discarding volume and much of the noise we don't care about.
  2. Compare two MFCC sequences with Dynamic Time Warping (DTW), which lines
     them up even if one is played a little faster or slower.
  3. A small DTW distance means "same sound". A threshold decides the cutoff.

Kept dependency-light (numpy + scipy + python_speech_features) and readable
on purpose -- this is the teaching core, and it has to run on a Raspberry Pi.
"""

import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import cdist
from python_speech_features import mfcc

SAMPLE_RATE = 16000

# DTW distance cutoff for a match. Measured on manually-recorded clips:
# genuine takes scored 12.8-17.7, impostors 38.6+, so ~25 sits in the gap,
# biased toward the positives (a false *accept* is the costlier error for a
# lock). NOTE: this was calibrated on hand-recorded clips; segmented live
# audio is a different capture path and may need re-measurement.
THRESHOLD = 25.0


def load_wav(path):
    """Read a mono 16 kHz WAV as a float array in roughly [-1, 1]."""
    sr, signal = wavfile.read(path)
    if signal.ndim > 1:            # stereo -> mono, just in case
        signal = signal.mean(axis=1)
    dtype = signal.dtype
    signal = signal.astype(np.float64)
    if np.issubdtype(dtype, np.integer):   # scale integer PCM to [-1, 1)
        signal /= float(np.iinfo(dtype).max) + 1.0
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    return signal


def trim_silence(signal, frame_len=320, keep_db=30.0):
    """Drop leading/trailing silence so we compare actual sound, not padding.

    Splits into short frames, keeps everything from the first to the last frame
    whose energy is within `keep_db` dB of the loudest frame.
    """
    n_frames = len(signal) // frame_len
    if n_frames < 2:
        return signal
    frames = signal[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.log((frames ** 2).mean(axis=1) + 1e-10)
    loud = energy.max()
    voiced = np.where(energy > loud - keep_db / 10.0)[0]
    if len(voiced) == 0:
        return signal
    start = voiced[0] * frame_len
    end = (voiced[-1] + 1) * frame_len
    return signal[start:end]


def extract_features(signal):
    """Turn a raw signal into a (frames x 13) MFCC matrix, normalized.

    Amplitude-normalize so loudness doesn't matter, then subtract the per-
    coefficient mean (cepstral mean normalization) to blunt mic/channel effects.
    """
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    signal = trim_silence(signal)
    feat = mfcc(signal, samplerate=SAMPLE_RATE, numcep=13, nfft=512)
    feat = feat - feat.mean(axis=0)
    return feat


def dtw_distance(a, b):
    """Length-normalized DTW distance between two MFCC sequences.

    Lower = more similar. Normalizing by path length keeps clips of different
    durations comparable.
    """
    local = cdist(a, b, metric="euclidean")   # (n x m) frame-to-frame costs
    n, m = local.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = local[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    return acc[n, m] / (n + m)


def features_for(path):
    """Convenience: path -> feature matrix."""
    return extract_features(load_wav(path))


def match(feat, sentinels, threshold=THRESHOLD):
    """Compare features against the enrolled set.

    Returns (is_match, distance) where distance is the DTW distance to the
    *nearest* sentinel. is_match is True when that nearest distance is within
    `threshold`.
    """
    dist = min(dtw_distance(feat, s) for s in sentinels)
    return dist <= threshold, dist
