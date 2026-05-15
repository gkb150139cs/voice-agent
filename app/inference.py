"""Age/gender inference (stub without torch; local uses torch_backend)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _years_from_logit(logit_age: float) -> float:
    """
    Map scalar age head to approximate years.
    The public HF card shows near-silence inputs around ~0.34 on the age channel; we treat
    moderate logits as already-normalized age fractions in [0, 1] and fall back to sigmoid
    for larger magnitudes.
    """
    if -0.05 <= logit_age <= 1.05:
        return float(np.clip(logit_age, 0.0, 1.0) * 100.0)
    import torch

    return float(torch.sigmoid(torch.tensor(logit_age, dtype=torch.float32)).item() * 100.0)


def _age_bracket(years: float) -> Tuple[str, float]:
    if years < 12:
        return "unknown", 0.35
    if years < 18:
        return "18-30", 0.45
    if years <= 30:
        return "18-30", 0.7
    if years <= 45:
        return "31-45", 0.7
    if years <= 60:
        return "46-60", 0.7
    return "60+", 0.7


def _gender_from_probs(p_female: float, p_male: float, p_child: float) -> Tuple[str, float]:
    probs = {"female": p_female, "male": p_male, "child": p_child}
    label = max(probs, key=probs.get)  # type: ignore[arg-type]
    conf = float(probs[label])
    if label == "child":
        return "unknown", float(max(p_female, p_male, 0.05))
    return label, conf


@dataclass
class InferenceEngine:
    model_id: str
    device: str
    mode: str  # local | stub

    _processor: Optional[Any] = None
    _model: Optional[Any] = None

    def load(self) -> None:
        if self.mode == "stub":
            logger.info("Inference engine in stub mode (no weights loaded).")
            return
        logger.info("Loading model %s on %s …", self.model_id, self.device)
        from app.torch_backend import load_model

        self._processor, self._model = load_model(self.model_id, self.device)
        logger.info("Model ready.")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    def predict(self, y: np.ndarray, sr: int, audio_quality: str) -> Dict[str, Any]:
        if y.size == 0:
            return self._unknown(audio_quality)

        if self.mode == "stub":
            rms = float(np.sqrt(np.mean(np.square(y))) + 1e-9)
            g = "male" if rms > 0.05 else "female"
            age = "31-45"
            g_conf = 0.62 + min(0.2, rms)
            a_conf = 0.55
            if audio_quality == "insufficient":
                return {
                    "gender": ("unknown", 0.22),
                    "age_bracket": ("unknown", 0.2),
                }
            if audio_quality == "degraded":
                g_conf *= 0.75
                a_conf *= 0.75
            return {"gender": (g, float(min(0.99, g_conf))), "age_bracket": (age, float(min(0.99, a_conf)))}

        assert self._processor is not None and self._model is not None
        from app.torch_backend import run_forward

        la, p_female, p_male, p_child = run_forward(self._processor, self._model, y, sr, self.device)

        years = _years_from_logit(la)
        age_label, age_conf_base = _age_bracket(years)
        g_label, g_conf = _gender_from_probs(p_female, p_male, p_child)

        if audio_quality == "insufficient":
            return {
                "gender": ("unknown", min(0.35, g_conf)),
                "age_bracket": ("unknown", min(0.35, age_conf_base)),
            }
        if audio_quality == "degraded":
            scale = 0.75
            return {
                "gender": (g_label if g_label != "unknown" else "unknown", min(0.99, g_conf * scale)),
                "age_bracket": (
                    age_label if age_label != "unknown" else "unknown",
                    min(0.99, age_conf_base * scale),
                ),
            }

        return {
            "gender": (g_label, float(min(0.99, g_conf))),
            "age_bracket": (age_label, float(min(0.99, age_conf_base))),
        }

    @staticmethod
    def _unknown(audio_quality: str) -> Dict[str, Any]:
        conf = 0.2 if audio_quality == "insufficient" else 0.25
        return {"gender": ("unknown", conf), "age_bracket": ("unknown", conf)}


def build_engine(settings: Any) -> InferenceEngine:
    mode = settings.inference_mode
    if mode not in ("local", "stub", "worker"):
        mode = "local"
    if mode == "worker" or settings.use_inference_worker:
        return InferenceEngine(model_id=settings.model_id, device=settings.torch_device, mode="stub")
    real_mode = "stub" if mode == "stub" else "local"
    return InferenceEngine(model_id=settings.model_id, device=settings.torch_device, mode=real_mode)
