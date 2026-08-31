from dataclasses import dataclass

from app.core.config import get_settings
from app.services.llm.base import LLMError, LLMService
from app.services.llm.mock import MockLLMService
from app.services.llm.openai_compatible import OpenAICompatibleLLMService


@dataclass(frozen=True)
class ProviderPreset:
    """Default endpoint and model for an OpenAI-compatible provider."""

    base_url: str
    model: str


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        base_url="https://api.openai.com/v1", model="gpt-4o-mini"
    ),
    "deepseek": ProviderPreset(
        base_url="https://api.deepseek.com", model="deepseek-chat"
    ),
    "qwen": ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
    ),
}


def get_llm_service() -> LLMService:
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLMService()
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Available providers: mock, "
            f"{', '.join(sorted(PROVIDER_PRESETS))}."
        )
    if not settings.llm_api_key:
        raise ValueError(
            f"LLM_API_KEY must be set to use the '{provider}' provider."
        )
    return OpenAICompatibleLLMService(
        api_key=settings.llm_api_key,
        model=settings.llm_model or preset.model,
        base_url=settings.llm_base_url or preset.base_url,
        timeout=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        max_output_retries=settings.llm_max_output_retries,
    )
