from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_path: Path = Path("news.sqlite3")
    sources_config_path: Path = Path("configs/sources.yaml")
    tickers_config_path: Path = Path("configs/tickers.yaml")
    request_timeout_seconds: float = Field(default=20.0, gt=0.0)
    news_scheduler_enabled: bool = True
    bootstrap_enabled: bool = True
    bootstrap_lookback_days: int = Field(default=1, ge=1)
    bootstrap_max_items_per_source: int = Field(default=50, ge=3)
    bootstrap_fallback_items: int = Field(default=3, ge=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
