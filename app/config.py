from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "voice-attribute-service"
    environment: str = Field(default="development", description="development|staging|production")
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000

    # Inference: stub (CI/smoke), local (in-process torch), worker (Redis + arq worker)
    inference_mode: str = Field(default="local", validation_alias="INFERENCE_MODE")
    model_id: str = Field(
        default="audeering/wav2vec2-large-robust-24-ft-age-gender",
        validation_alias="MODEL_ID",
    )
    torch_device: str = Field(default="cpu", validation_alias="TORCH_DEVICE")

    # When true, POST /analyze enqueues to arq and waits for worker result
    use_inference_worker: bool = Field(default=False, validation_alias="USE_INFERENCE_WORKER")
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    inference_job_timeout_s: float = Field(default=120.0, validation_alias="INFERENCE_JOB_TIMEOUT_S")

    # Audio
    target_sample_rate: int = Field(default=16000, validation_alias="TARGET_SAMPLE_RATE")
    min_speech_seconds: float = Field(default=0.4, validation_alias="MIN_SPEECH_SECONDS")
    degraded_snr_db: float = Field(default=8.0, validation_alias="DEGRADED_SNR_DB")
    clipping_ratio_threshold: float = Field(default=0.002, validation_alias="CLIPPING_RATIO_THRESHOLD")

    # Privacy / ops
    max_upload_bytes: int = Field(default=5_000_000, validation_alias="MAX_UPLOAD_BYTES")
    request_id_header: str = Field(default="X-Request-ID", validation_alias="REQUEST_ID_HEADER")

    # Optional bonus
    enable_lang_field: bool = Field(default=False, validation_alias="ENABLE_LANG_FIELD")


@lru_cache
def get_settings() -> Settings:
    return Settings()
