"""
Configuration — loads all settings from environment variables.
Never hardcode secrets. All values have safe defaults for local dev.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_timeout_seconds: int = 30

    # ── Input Guards ─────────────────────────────────────────────────────────
    max_instruction_chars: int = 2_000       # ~500 tokens
    max_document_chars: int = 40_000         # ~10k tokens — adjust per plan
    max_combined_chars: int = 42_000

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 2
    retry_delay_seconds: float = 1.0

    # ── App ───────────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()