"""arq worker: GPU/CPU inference off the API hot path."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from app.config import Settings
from app.inference import InferenceEngine
from app.pipeline import analyze_wav_job_bytes
from app.redis_url import redis_settings_from_url

logger = logging.getLogger(__name__)


async def infer_wav_task(ctx: Dict[str, Any], wav_bytes: bytes) -> Dict[str, Any]:
    engine: InferenceEngine = ctx["engine"]
    settings: Settings = ctx["settings"]
    return analyze_wav_job_bytes(wav_bytes, settings, engine)


async def startup(ctx: Dict[str, Any]) -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    s = Settings()
    # Worker always runs real local inference; API may use INFERENCE_MODE=worker only.
    mode = "stub" if s.inference_mode == "stub" else "local"
    eng = InferenceEngine(model_id=s.model_id, device=s.torch_device, mode=mode)
    eng.load()
    ctx["engine"] = eng
    ctx["settings"] = s
    logger.info("arq worker ready (torch_device=%s, model_id=%s)", s.torch_device, s.model_id)


async def shutdown(ctx: Dict[str, Any]) -> None:
    eng = ctx.get("engine")
    if isinstance(eng, InferenceEngine):
        eng.unload()
    logger.info("arq worker stopped.")


class WorkerSettings:
    functions = [infer_wav_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = redis_settings_from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
