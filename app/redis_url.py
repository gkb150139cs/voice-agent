"""Parse REDIS_URL into arq RedisSettings (no from_dsn version coupling)."""

from __future__ import annotations

from urllib.parse import urlparse

from arq.connections import RedisSettings


def redis_settings_from_url(url: str) -> RedisSettings:
    parsed = urlparse(url)
    path = (parsed.path or "").lstrip("/")
    database = int(path) if path.isdigit() else 0
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=database,
        password=parsed.password,
        username=parsed.username,
    )
