#!/usr/bin/env python3
"""Write samples/tone.wav (1.2s, 16 kHz mono PCM16) for manual smoke tests.

Uses only the stdlib so ``python3 samples/generate_tone.py`` works without numpy/soundfile.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def main() -> None:
    out = Path(__file__).resolve().parent / "tone.wav"
    sr = 16_000
    duration_s = 1.2
    freq_hz = 440.0
    amplitude = 0.15  # fraction of int16 full scale

    n = int(sr * duration_s)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            s = amplitude * 32767.0 * math.sin(2.0 * math.pi * freq_hz * (i / sr))
            s_i = int(max(-32768, min(32767, round(s))))
            wf.writeframes(struct.pack("<h", s_i))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
