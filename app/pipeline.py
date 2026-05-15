"""End-to-end analyze pipeline (RAM-only; no persistent audio)."""

from __future__ import annotations

import io
import logging
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from app.audio_preprocess import AudioCodec, band_limit_denoise_simple, decode_audio_bytes
from app.audio_quality import compute_quality
from app.config import Settings
from app.inference import InferenceEngine
from app.schemas import AnalyzeResponse, AttributePrediction, LanguageGuess
from app.vad import build_speech_mask

logger = logging.getLogger(__name__)


def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def analyze_from_array(
    y: np.ndarray,
    sr: int,
    *,
    contact_id: str,
    settings: Settings,
    engine: InferenceEngine,
) -> AnalyzeResponse:
    t0 = time.perf_counter()
    if y.size == 0:
        audio_quality = "insufficient"
        preds = engine.predict(y, sr, audio_quality)
    else:
        y = band_limit_denoise_simple(y, sr)
        mask = build_speech_mask(y, sr, prefer_silero=True, torch_device=settings.torch_device)
        speech_audio = y.copy()
        if mask.size == y.size and mask.any():
            speech_audio = y * mask.astype(np.float32)
        audio_quality, _metrics = compute_quality(
            y,
            sr,
            mask,
            degraded_snr_db=settings.degraded_snr_db,
            clipping_threshold=settings.clipping_ratio_threshold,
            min_speech_seconds=settings.min_speech_seconds,
        )
        preds = engine.predict(speech_audio if np.any(mask) else y, sr, audio_quality)

    g_pred, g_conf = preds["gender"]
    a_pred, a_conf = preds["age_bracket"]

    lang: Optional[LanguageGuess] = None
    if settings.enable_lang_field:
        lang = LanguageGuess(prediction=None, confidence=0.0)

    processing_ms = int((time.perf_counter() - t0) * 1000)
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=AttributePrediction(prediction=g_pred, confidence=float(g_conf)),
        age_bracket=AttributePrediction(prediction=a_pred, confidence=float(a_conf)),
        processing_ms=processing_ms,
        audio_quality=audio_quality,  # type: ignore[arg-type]
        language=lang,
    )


def analyze_upload(
    raw: bytes,
    *,
    codec: AudioCodec,
    source_sr: int,
    contact_id: Optional[str],
    settings: Settings,
    engine: InferenceEngine,
) -> AnalyzeResponse:
    cid = contact_id or str(uuid.uuid4())
    y, sr = decode_audio_bytes(raw, codec=codec, sample_rate=source_sr, target_sample_rate=settings.target_sample_rate)
    return analyze_from_array(y, sr, contact_id=cid, settings=settings, engine=engine)


def analyze_wav_job_bytes(wav_bytes: bytes, settings: Settings, engine: InferenceEngine) -> Dict[str, Any]:
    """Used by arq worker: expects PCM wav bytes @ any rate, mono/stereo."""
    buf = io.BytesIO(wav_bytes)
    y, sr = sf.read(buf, dtype="float32", always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim == 2:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if sr != settings.target_sample_rate:
        import librosa

        y = librosa.resample(y, orig_sr=int(sr), target_sr=settings.target_sample_rate).astype(np.float32)
        sr = settings.target_sample_rate
    # contact_id is re-generated at API layer for worker responses
    out = analyze_from_array(y, sr, contact_id="pending", settings=settings, engine=engine)
    return out.model_dump()
