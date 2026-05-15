"""Audio quality heuristics: SNR proxy, clipping, speech coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class QualityMetrics:
    snr_db: float
    speech_seconds: float
    clipping_ratio: float


def estimate_snr_db(y: np.ndarray, sr: int) -> float:
    """Very rough SNR estimate using high-energy frames as signal vs low-energy as noise."""
    if y.size == 0 or sr <= 0:
        return -80.0
    frame = max(1, int(0.025 * sr))
    hop = max(1, frame // 2)
    energies = []
    for i in range(0, len(y) - frame, hop):
        e = float(np.mean(np.square(y[i : i + frame])))
        energies.append(e)
    if not energies:
        return -80.0
    e = np.array(energies, dtype=np.float64)
    e_sorted = np.sort(e)
    noise = float(np.mean(e_sorted[: max(1, len(e_sorted) // 10)])) + 1e-12
    signal = float(np.mean(e_sorted[max(1, int(0.9 * len(e_sorted))) :])) + 1e-12
    snr = 10.0 * np.log10(signal / noise)
    return float(np.clip(snr, -20.0, 60.0))


def clipping_ratio(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) > 0.995))


def speech_seconds_from_mask(mask: np.ndarray, sr: int) -> float:
    if sr <= 0 or mask.size == 0:
        return 0.0
    return float(np.sum(mask.astype(np.float32)) / sr)


def compute_quality(
    y: np.ndarray,
    sr: int,
    speech_mask: np.ndarray,
    *,
    degraded_snr_db: float,
    clipping_threshold: float,
    min_speech_seconds: float,
) -> Tuple[str, QualityMetrics]:
    snr = estimate_snr_db(y, sr)
    clip = clipping_ratio(y)
    sp = speech_seconds_from_mask(speech_mask, sr)
    metrics = QualityMetrics(snr_db=snr, speech_seconds=sp, clipping_ratio=clip)

    if sp < min_speech_seconds:
        return "insufficient", metrics

    if snr < degraded_snr_db or clip > clipping_threshold:
        return "degraded", metrics

    return "good", metrics
