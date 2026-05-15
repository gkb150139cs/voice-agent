"""Energy-based VAD fallback; optional Silero when torch is available."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def energy_vad_mask(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame = max(1, int(frame_ms * sr / 1000.0))
    hop = max(1, int(hop_ms * sr / 1000.0))
    mask = np.zeros(len(y), dtype=np.bool_)
    energies = []
    for start in range(0, len(y) - frame, hop):
        e = float(np.sqrt(np.mean(np.square(y[start : start + frame]))) + 1e-12)
        energies.append((start, e))
    if not energies:
        return mask
    e_arr = np.array([e for _, e in energies], dtype=np.float64)
    if e_arr.size == 0:
        return mask
    thresh = float(np.percentile(e_arr, 35) * 2.2)
    for start, e in energies:
        if e >= thresh:
            mask[start : start + frame] = True
    return mask


def silero_vad_mask(y: np.ndarray, sr: int, device: str = "cpu") -> Optional[np.ndarray]:
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero_vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        (get_speech_timestamps, _save_audio, _read_audio, _VADIterator, _collect_chunks) = utils
        if sr != 16000:
            import librosa

            y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000).astype(np.float32)
            work_sr = 16000
        else:
            y16 = y.astype(np.float32)
            work_sr = 16000
        wav = torch.from_numpy(y16).to(device)
        ts = get_speech_timestamps(wav, model, sampling_rate=work_sr)
        mask = np.zeros(len(y), dtype=np.bool_)
        if work_sr != sr:
            ratio = sr / work_sr
            for t in ts:
                a = int(t["start"] * ratio)
                b = int(t["end"] * ratio)
                a = max(0, min(a, len(mask) - 1))
                b = max(0, min(b, len(mask)))
                if b > a:
                    mask[a:b] = True
        else:
            for t in ts:
                a, b = int(t["start"]), int(t["end"])
                if b > a:
                    mask[a:b] = True
        return mask
    except Exception:
        return None


def build_speech_mask(y: np.ndarray, sr: int, prefer_silero: bool = True, torch_device: str = "cpu") -> np.ndarray:
    if y.size == 0:
        return np.zeros(0, dtype=np.bool_)
    if os.environ.get("USE_SILERO_VAD", "true").lower() in ("0", "false", "no"):
        return energy_vad_mask(y, sr)
    if prefer_silero:
        m = silero_vad_mask(y, sr, device=torch_device)
        if m is not None and m.any():
            return m
    return energy_vad_mask(y, sr)
