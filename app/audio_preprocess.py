"""Decode telephony-style audio to mono float32 @ target sample rate (no disk I/O)."""

from __future__ import annotations

import audioop
import io
import shutil
from enum import Enum
from typing import Tuple

import numpy as np
import soundfile as sf


class AudioCodec(str, Enum):
    wav = "wav"
    mulaw = "mulaw"
    pcm_s16le = "pcm_s16le"
    auto = "auto"


def _ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None or shutil.which("ffmpeg") is not None


def _to_mono_float32(y: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if y.ndim > 1:
        y = np.mean(y.astype(np.float64), axis=1).astype(np.float32)
    else:
        y = y.astype(np.float32)
    if sr != target_sr:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    return y, sr


def decode_audio_bytes(
    data: bytes,
    *,
    codec: AudioCodec,
    sample_rate: int,
    target_sample_rate: int,
) -> Tuple[np.ndarray, int]:
    """
    Returns (mono float32 waveform, sample_rate).
    """
    if not data:
        return np.zeros(0, dtype=np.float32), target_sample_rate

    if codec == AudioCodec.mulaw:
        pcm = audioop.ulaw2lin(data, 2)
        y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return _to_mono_float32(y, sample_rate, target_sample_rate)

    if codec == AudioCodec.pcm_s16le:
        n = len(data) - (len(data) % 2)
        if n < 2:
            return np.zeros(0, dtype=np.float32), target_sample_rate
        y = np.frombuffer(data[:n], dtype=np.int16).astype(np.float32) / 32768.0
        return _to_mono_float32(y, sample_rate, target_sample_rate)

    if codec == AudioCodec.wav:
        buf = io.BytesIO(data)
        y, sr = sf.read(buf, always_2d=False, dtype="float32")
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = np.mean(y, axis=1)
        return _to_mono_float32(y.astype(np.float32), int(sr), target_sample_rate)

    if codec == AudioCodec.auto:
        # Prefer libsndfile (no ffmpeg): WAV, FLAC, OGG, CAF, etc.
        try:
            buf = io.BytesIO(data)
            y, sr = sf.read(buf, always_2d=False, dtype="float32")
            if isinstance(y, np.ndarray) and y.size > 0:
                if y.ndim == 2:
                    y = np.mean(y, axis=1)
                return _to_mono_float32(y.astype(np.float32), int(sr), target_sample_rate)
        except Exception:
            pass

        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return decode_audio_bytes(
                data, codec=AudioCodec.wav, sample_rate=sample_rate, target_sample_rate=target_sample_rate
            )

        if _ffprobe_available():
            try:
                from pydub import AudioSegment

                seg = AudioSegment.from_file(io.BytesIO(data))
                sr = seg.frame_rate
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                if seg.channels > 1:
                    samples = samples.reshape((-1, seg.channels)).mean(axis=1)
                max_val = float(1 << (8 * seg.sample_width - 1))
                y = (samples / max_val).astype(np.float32)
                return _to_mono_float32(y, int(sr), target_sample_rate)
            except Exception:
                pass

        # Last resort: assume raw s16le telephony (even byte length only).
        if len(data) >= 2 and len(data) % 2 == 0:
            return decode_audio_bytes(
                data,
                codec=AudioCodec.pcm_s16le,
                sample_rate=sample_rate,
                target_sample_rate=target_sample_rate,
            )

        raise ValueError(
            "Could not decode audio in auto mode. Upload WAV/FLAC/OGG (libsndfile), "
            "set codec=mulaw or pcm_s16le for raw telephony, or install ffmpeg/ffprobe for MP3/WebM."
        )

    raise ValueError(f"Unsupported codec: {codec}")


def band_limit_denoise_simple(y: np.ndarray, sr: int) -> np.ndarray:
    """Lightweight high-pass + gentle noise gate for telephony noise."""
    if y.size == 0:
        return y
    from scipy import signal

    sos = signal.butter(2, 80, btype="highpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, y).astype(np.float32)
    frame = int(0.02 * sr)
    if frame < 8:
        return y
    rms = []
    for i in range(0, len(y) - frame, frame):
        rms.append(float(np.sqrt(np.mean(np.square(y[i : i + frame]))) + 1e-9))
    if not rms:
        return y
    noise_floor = float(np.percentile(np.array(rms), 10))
    gate = max(1e-4, noise_floor * 1.8)
    mask = np.ones_like(y)
    for i in range(0, len(y) - frame, frame):
        seg_rms = float(np.sqrt(np.mean(np.square(y[i : i + frame]))) + 1e-9)
        if seg_rms < gate:
            mask[i : i + frame] *= 0.15
    out = (y * mask).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 0:
        out = (out / peak * min(1.0, peak)).astype(np.float32)
    return out
