from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    database_url: str = Field(
        default="postgresql+psycopg://ml_eval:ml_eval@localhost:5432/ml_eval",
        validation_alias="DATABASE_URL",
    )
    auto_create_tables: bool = Field(default=True, validation_alias="AUTO_CREATE_TABLES")
    batch_retry_attempts: int = Field(
        default=3, ge=1, le=8, validation_alias="BATCH_RETRY_ATTEMPTS"
    )
    batch_retry_backoff_seconds: float = Field(
        default=0.05,
        ge=0,
        le=5,
        validation_alias="BATCH_RETRY_BACKOFF_SECONDS",
    )
    store_predictions: bool = Field(default=False, validation_alias="STORE_PREDICTIONS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
