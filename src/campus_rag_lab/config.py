from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    gemini_model: str = "gemini-1.5-flash"
    vector_backend: str = "local"
    pinecone_api_key: str | None = None
    rate_limit_sleep: float = 1.0
    cache_threshold: float = 0.82


def load_settings(env_file: str | Path | None = None) -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        vector_backend=os.getenv("VECTOR_BACKEND", "local"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        rate_limit_sleep=float(os.getenv("RATE_LIMIT_SLEEP", "1.0")),
        cache_threshold=float(os.getenv("CACHE_THRESHOLD", "0.82")),
    )

