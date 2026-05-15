#!/usr/bin/env python3
"""Write samples/tone.wav (1.2s, 16 kHz) for manual smoke tests."""

from __future__ import annotations

import pathlib

import numpy as np
import soundfile as sf


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    out = root / "tone.wav"
    sr = 16000
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    y = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(out, y, sr, subtype="PCM_16")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
