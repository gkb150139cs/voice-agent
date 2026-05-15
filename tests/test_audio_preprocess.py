import io

import numpy as np
import soundfile as sf

from app.audio_preprocess import AudioCodec, decode_audio_bytes


def test_auto_decodes_wav_without_pydub_ffmpeg() -> None:
    sr = 16000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    y = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    raw = buf.getvalue()
    y2, sr2 = decode_audio_bytes(
        raw,
        codec=AudioCodec.auto,
        sample_rate=8000,
        target_sample_rate=16000,
    )
    assert y2.size > 0
    assert sr2 == 16000


def test_pcm_s16le_truncates_odd_byte_count() -> None:
    raw = b"\x00\x00\x01"  # 3 bytes -> first 2 used as one int16 sample
    y, sr = decode_audio_bytes(
        raw,
        codec=AudioCodec.pcm_s16le,
        sample_rate=8000,
        target_sample_rate=16000,
    )
    assert y.size >= 1
    assert sr == 16000
