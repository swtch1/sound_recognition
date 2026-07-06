"""Keypress-controlled recorder: Enter to start, Enter to stop.

    python record.py sentinels/sentinel_02.wav          # default mic
    python record.py negatives/neg_04.wav 1             # specific device index

Writes mono 16 kHz 16-bit WAV -- the format the matcher expects. Same capture
path we'll reuse for live listening on the Pi. Run it directly in your own
terminal (it waits on Enter, so it needs an interactive prompt).
"""

import sys
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


def record(path, device=None):
    chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    input("Press Enter to START recording...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        device=device, callback=callback):
        input(">>> recording -- press Enter to STOP <<<")

    if not chunks:
        print("No audio captured.", file=sys.stderr)
        return
    audio = np.concatenate(chunks).reshape(-1)

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())

    dur = len(audio) / SAMPLE_RATE
    peak = np.abs(audio).max() / 32768.0
    warn = "  <-- very quiet, check the mic" if peak < 0.02 else ""
    print(f"wrote {path}: {dur:.2f}s, peak {peak:.0%}{warn}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python record.py <output.wav> [device_index]")
    dev = int(sys.argv[2]) if len(sys.argv) > 2 else None
    record(sys.argv[1], dev)
