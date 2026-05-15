"""FastAPI entrypoint: REST + WebSocket + optional Redis-backed inference."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from arq import create_pool
from arq.connections import ArqRedis
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse

from app.audio_preprocess import AudioCodec, decode_audio_bytes
from app.config import Settings, get_settings
from app.inference import InferenceEngine, build_engine
from app.pipeline import analyze_from_array, analyze_upload, to_wav_bytes
from app.redis_url import redis_settings_from_url
from app.schemas import AnalyzeResponse, StreamPartialResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=get_settings().log_level)
    settings = get_settings()
    engine = build_engine(settings)
    if not settings.use_inference_worker and settings.inference_mode != "worker":
        engine.load()
    pool: Optional[ArqRedis] = None
    if settings.use_inference_worker:
        pool = await create_pool(redis_settings_from_url(settings.redis_url))
    app.state.settings = settings
    app.state.engine = engine
    app.state.redis_pool = pool
    yield
    if pool is not None:
        await pool.close()
        app.state.redis_pool = None
    engine.unload()


app = FastAPI(title="Voice Attribute Service", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    settings: Settings = request.app.state.settings
    rid = request.headers.get("x-request-id")
    if not rid:
        rid = str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", extra={"request_id": rid})
        raise
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - t0) * 1000))
    return response


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    if settings.use_inference_worker:
        pool: Optional[ArqRedis] = request.app.state.redis_pool
        if pool is None:
            return JSONResponse({"ready": False, "reason": "redis_pool_missing"}, status_code=503)
        try:
            pong = await pool.ping()
            if not pong:
                return JSONResponse({"ready": False, "reason": "redis_ping_failed"}, status_code=503)
        except Exception as e:
            logger.warning("readyz_redis_failed: %s", e)
            return JSONResponse({"ready": False, "reason": "redis_unreachable"}, status_code=503)
    return JSONResponse({"ready": True})


@app.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True)
async def analyze(
    request: Request,
    audio: UploadFile = File(..., description="Raw audio (WAV, mulaw, or auto-detect)."),
    contact_id: Optional[str] = Form(None),
    codec: str = Form("auto"),
    sample_rate: int = Form(8000, description="Source sample rate for raw codecs (e.g. 8000 telephony)."),
) -> AnalyzeResponse:
    settings: Settings = request.app.state.settings
    engine: InferenceEngine = request.app.state.engine
    raw = await audio.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="audio too large")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty audio")

    try:
        ac = AudioCodec(codec)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    cid = contact_id or str(uuid.uuid4())
    rid = getattr(request.state, "request_id", None)
    logger.info(
        "analyze_start",
        extra={"request_id": rid, "contact_id": cid, "bytes": len(raw), "codec": codec},
    )

    try:
        if settings.use_inference_worker and request.app.state.redis_pool is not None:
            y, sr = decode_audio_bytes(
                raw,
                codec=ac,
                sample_rate=sample_rate,
                target_sample_rate=settings.target_sample_rate,
            )
            job = await request.app.state.redis_pool.enqueue_job("infer_wav_task", to_wav_bytes(y, sr))
            data = await job.result(timeout=settings.inference_job_timeout_s)
            data["contact_id"] = cid
            return AnalyzeResponse.model_validate(data)

        return analyze_upload(
            raw,
            codec=ac,
            source_sr=sample_rate,
            contact_id=cid,
            settings=settings,
            engine=engine,
        )
    except asyncio.TimeoutError as e:
        raise HTTPException(status_code=504, detail="inference worker timeout") from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_failed", extra={"request_id": rid, "contact_id": cid})
        raise HTTPException(status_code=500, detail="inference error") from e


@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket) -> None:
    await ws.accept()
    settings: Settings = ws.app.state.settings
    engine: InferenceEngine = ws.app.state.engine
    pool: Optional[ArqRedis] = ws.app.state.redis_pool

    codec_str = ws.query_params.get("codec", "auto")
    sample_rate = int(ws.query_params.get("sample_rate", "8000"))
    contact_id = ws.query_params.get("contact_id") or str(uuid.uuid4())
    try:
        ac = AudioCodec(codec_str)
    except ValueError:
        await ws.close(code=4400)
        return

    buf = bytearray()
    last_emit = 0.0
    min_emit_interval_s = 0.85

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                buf.extend(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                if msg["text"].strip().lower() == "reset":
                    buf.clear()

            if len(buf) > settings.max_upload_bytes:
                await ws.send_json({"error": "buffer_overflow"})
                buf.clear()
                continue

            min_bytes = sample_rate if ac == AudioCodec.mulaw else sample_rate * 2
            if len(buf) < min_bytes:
                continue

            now = time.perf_counter()
            raw = bytes(buf)
            try:
                y, sr = decode_audio_bytes(
                    raw,
                    codec=ac,
                    sample_rate=sample_rate,
                    target_sample_rate=settings.target_sample_rate,
                )
            except ValueError as e:
                await ws.send_json({"error": "decode_failed", "detail": str(e)})
                continue
            if y.size < int(sr * 1.0):
                continue
            if now - last_emit < min_emit_interval_s:
                continue

            if settings.use_inference_worker and pool is not None:
                job = await pool.enqueue_job("infer_wav_task", to_wav_bytes(y, sr))
                data = await job.result(timeout=settings.inference_job_timeout_s)
                partial = AnalyzeResponse.model_validate(data)
            else:
                partial = analyze_from_array(y, sr, contact_id=contact_id, settings=settings, engine=engine)

            out = StreamPartialResponse(
                contact_id=contact_id,
                gender=partial.gender,
                age_bracket=partial.age_bracket,
                audio_quality=partial.audio_quality,
                window_seconds=float(y.size / sr),
            )
            await ws.send_json(out.model_dump(exclude_none=True))
            last_emit = now
    except WebSocketDisconnect:
        return
    finally:
        buf.clear()
