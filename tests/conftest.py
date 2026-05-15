import os

# Default test profile: no GPU weights, no Redis worker.
os.environ.setdefault("INFERENCE_MODE", "stub")
os.environ.setdefault("USE_INFERENCE_WORKER", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("USE_SILERO_VAD", "false")
