# Voice Attribute Service

FastAPI backend that ingests short telephony-style audio, estimates **gender** and **age bracket** with calibrated confidences, and emits an **audio quality** signal for noisy logistics calls. Optional **Redis + arq** workers isolate GPU/CPU inference from the API process for horizontal scaling.

For local development, install **ffmpeg** (provides `ffprobe`) if you use Swagger to upload MP3/WebM or rely on pydub fallbacks: `sudo apt install ffmpeg` on Debian/Ubuntu.

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio  # add --index-url https://download.pytorch.org/whl/cpu for CPU-only wheels
pip install -r requirements.txt -r requirements-dev.txt
cp example.env .env  # optional; defaults match example.env
export INFERENCE_MODE=stub USE_SILERO_VAD=false  # fast smoke without HF weights
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run tests:

```bash
USE_SILERO_VAD=false pytest
```

## Docker Compose (API + Redis + worker)

```bash
docker compose up --build
```

Redis is **not** published to the host (only `api` port **8000** is), so it never collides with a local Redis on 6379/6380. API and worker still use `redis://redis:6379/0` on the Docker network. To open a shell on Redis: `docker compose exec redis redis-cli`.

`docker-compose.yml` forces `USE_INFERENCE_WORKER=true` and `INFERENCE_MODE=worker` on the API container so it stays lightweight while the `worker` service loads the Hugging Face model and executes `infer_wav_task` jobs. Tune `TORCH_DEVICE` and deploy multiple worker replicas for throughput.

### Smoke test (multipart)

Generate a sample clip:

```bash
python samples/generate_tone.py
curl -sS -X POST "http://localhost:8000/analyze" \
  -F "audio=@samples/tone.wav;type=audio/wav" \
  -F "codec=wav" \
  -F "sample_rate=16000" \
  -F "contact_id=demo-1" | jq
```

## API

- `POST /analyze` — multipart upload (`audio` file) with optional `contact_id`, `codec` (`auto|wav|mulaw|pcm_s16le`), and `sample_rate` (for raw codecs, default `8000`).
- `GET /healthz` — process up.
- `GET /readyz` — Redis reachable when `USE_INFERENCE_WORKER=true`.
- `WS /ws/analyze?codec=mulaw&sample_rate=8000&contact_id=...` — stream binary chunks; server emits progressive JSON when at least ~1s of decoded audio is available (throttled).

Response shape matches the assignment contract (`gender`, `age_bracket`, `processing_ms`, `audio_quality`).

## Privacy & data handling

- Audio bytes are processed **only in RAM** for the lifetime of a request/job; buffers are not written to disk and references are dropped after serialization for Redis workers.
- Logs include **metadata only** (request id, contact id, byte length, codec, timings). Raw audio is never logged.
- Set `LOG_LEVEL=WARNING` in production to reduce noise.

## Model choice

We fine-tune-ready **audeering/wav2vec2-large-robust-24-ft-age-gender** (single forward pass for age + gender logits) because it is trained on heterogeneous telephony/crowdsourced speech (Common Voice, VoxCeleb2, TIMIT, aGender) and exposes calibrated softmax gender probabilities plus a continuous age head that maps into the requested brackets. Trade-offs: the checkpoint is large (~0.3B parameters) and cold-starts are slow on CPU; for production we would quantize (ONNX INT8), cache warm workers, and pin GPU pools per AZ.

## Design note

The service separates **fast I/O** (FastAPI + multipart/WebSocket ingestion, resampling to 16 kHz mono, light denoise) from **heavy inference** (Wav2Vec2 forward pass). For scale, the API optionally enqueues PCM WAV payloads to **Redis** via **arq**, letting stateless API pods autoscale independently of memory-hungry workers. Workers deserialize in RAM, reuse a single loaded model per process, and return structured JSON, which keeps tail latency predictable under burst telephony traffic. Audio quality is computed with cheap signal metrics (band-limited SNR proxy, clipping rate, VAD-weighted speech seconds) *before* trusting classifier confidences: when quality is `insufficient` or `degraded`, predictions are damped or marked `unknown`, which matches logistics scenarios (highway noise, GSM artifacts). Progressive WebSocket updates re-run inference on a growing buffer so confidence rises as more speech arrives—mirroring the bonus scenario. With more time, I would add ONNXRuntime batching, per-language heads, and a calibration layer (temperature scaling) fit on Common Voice; for 1k concurrent calls I would shard Redis queues by tenant, run HPA on workers with GPU node pools, cap upload duration server-side, and add back-pressure when queue depth exceeds SLO.

## Bonus: evaluation harness

```bash
pip install datasets pandas tqdm torch torchaudio
python scripts/eval_common_voice.py --limit 200 --language en
```

## Known limitations

- CPU inference on the large Wav2Vec2 model may exceed the 500 ms target for long clips; use GPU workers, shorter windows, or ONNX for stricter SLOs.
- `audioop` µ-law decode uses the Python stdlib module (deprecated in 3.13); swap to `audioop-lts` or `g711` when upgrading interpreters.
- Language/accent detection is stubbed unless you extend the pipeline (field gated by `ENABLE_LANG_FIELD`).
