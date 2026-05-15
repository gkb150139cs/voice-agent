#!/usr/bin/env python3
"""
Optional evaluation harness against Mozilla Common Voice (gender labels).

Install extras first:
  pip install datasets pandas tqdm

Run:
  python scripts/eval_common_voice.py --split test --limit 200 --language en

This script is intentionally lightweight: it downloads clips, runs the same
`InferenceEngine` path as production, and prints simple accuracy / mean confidence.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import wave

import numpy as np


def _write_wav_int16(path: str, samples: np.ndarray, sr: int) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Please `pip install datasets pandas tqdm` to run this script.") from e

    from app.config import Settings
    from app.inference import InferenceEngine
    from app.pipeline import analyze_from_array

    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--language", default="en")
    p.add_argument("--model-id", default=os.environ.get("MODEL_ID", "audeering/wav2vec2-large-robust-24-ft-age-gender"))
    args = p.parse_args()

    os.environ["INFERENCE_MODE"] = "local"
    settings = Settings()
    eng = InferenceEngine(args.model_id, settings.torch_device, "local")
    eng.load()

    ds = load_dataset("mozilla-foundation/common_voice_16_1", args.language, split=args.split, streaming=True)
    correct = 0
    total = 0
    conf_sum = 0.0

    for row in ds:
        if total >= args.limit:
            break
        gender = row.get("gender")
        if gender not in ("male", "female"):
            continue
        arr = row["audio"]["array"]
        sr = int(row["audio"]["sampling_rate"])
        y = np.asarray(arr, dtype=np.float32)
        if y.size == 0:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            _write_wav_int16(tmp.name, y, sr)
            raw = open(tmp.name, "rb").read()
        # Reuse decode path via soundfile in analyze_upload would need file; call analyze_from_array:
        import soundfile as sf

        y16, sr2 = sf.read(io.BytesIO(raw), dtype="float32")
        if y16.ndim > 1:
            y16 = np.mean(y16, axis=1)
        if sr2 != settings.target_sample_rate:
            import librosa

            y16 = librosa.resample(y16.astype(np.float32), orig_sr=int(sr2), target_sr=settings.target_sample_rate)
            sr2 = settings.target_sample_rate
        out = analyze_from_array(y16.astype(np.float32), int(sr2), contact_id="eval", settings=settings, engine=eng)
        if out.audio_quality == "insufficient":
            continue
        pred = out.gender.prediction
        ok = pred == gender
        correct += int(ok)
        total += 1
        conf_sum += float(out.gender.confidence)

    if total == 0:
        print("No usable rows (check split/language or increase limit).")
        return
    print(f"evaluated_rows={total} accuracy={correct/total:.4f} mean_confidence={conf_sum/total:.4f}")


if __name__ == "__main__":
    main()
