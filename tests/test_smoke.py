import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


def test_healthz() -> None:
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_analyze_wav_stub() -> None:
    sr = 16000
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()

    with TestClient(app) as client:
        files = {"audio": ("chunk.wav", audio_bytes, "application/octet-stream")}
        data = {"codec": "wav", "sample_rate": "16000", "contact_id": "test-contact-1"}
        r = client.post("/analyze", files=files, data=data)
        assert r.status_code == 200, r.text
        js = r.json()
        assert js["contact_id"] == "test-contact-1"
        assert js["gender"]["prediction"] in ("male", "female", "unknown")
        assert 0.0 <= js["gender"]["confidence"] <= 1.0
        assert js["age_bracket"]["prediction"] in ("18-30", "31-45", "46-60", "60+", "unknown")
        assert js["audio_quality"] in ("good", "degraded", "insufficient")
        assert isinstance(js["processing_ms"], int)


def test_stub_insufficient_speech() -> None:
    from app.config import Settings
    from app.inference import InferenceEngine
    from app.pipeline import analyze_from_array

    sr = 16000
    y = np.zeros(int(sr * 0.05), dtype=np.float32)
    eng = InferenceEngine("dummy", "cpu", "stub")
    out = analyze_from_array(y, sr, contact_id="x", settings=Settings(), engine=eng)
    assert out.audio_quality == "insufficient"
    assert out.gender.prediction == "unknown"


def test_stub_loud_short_may_predict() -> None:
    from app.config import Settings
    from app.inference import InferenceEngine
    from app.pipeline import analyze_from_array

    sr = 16000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    eng = InferenceEngine("dummy", "cpu", "stub")
    out = analyze_from_array(y, sr, contact_id="x", settings=Settings(), engine=eng)
    assert out.gender.prediction in ("male", "female", "unknown")
