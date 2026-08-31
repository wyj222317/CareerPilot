from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "CareerPilot"
    app_env: str = "development"
    llm_provider: str = "mock"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout_seconds: float = 60.0
    llm_temperature: float = 0.1
    llm_max_output_retries: int = 2
    max_upload_size_mb: int = 10
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    embedding_provider: str = "mock"
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    evidence_top_k: int = 4

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
