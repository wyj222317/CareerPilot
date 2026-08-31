import pytest

import app.services.llm as llm_package
from app.core.config import Settings
from app.services.llm import PROVIDER_PRESETS, MockLLMService, OpenAICompatibleLLMService


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setattr(
        llm_package, "get_settings", lambda: Settings(llm_provider="anthropic")
    )
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm_package.get_llm_service()


def test_mock_provider_is_default(monkeypatch):
    monkeypatch.setattr(
        llm_package, "get_settings", lambda: Settings(llm_provider="mock")
    )
    assert isinstance(llm_package.get_llm_service(), MockLLMService)


def test_real_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        llm_package,
        "get_settings",
        lambda: Settings(llm_provider="openai", llm_api_key=""),
    )
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        llm_package.get_llm_service()


def test_deepseek_uses_provider_preset(monkeypatch):
    monkeypatch.setattr(
        llm_package,
        "get_settings",
        lambda: Settings(llm_provider="deepseek", llm_api_key="test-api-key"),
    )
    service = llm_package.get_llm_service()
    assert isinstance(service, OpenAICompatibleLLMService)
    assert service.model == PROVIDER_PRESETS["deepseek"].model
    assert service.base_url == PROVIDER_PRESETS["deepseek"].base_url


def test_model_and_base_url_overrides(monkeypatch):
    monkeypatch.setattr(
        llm_package,
        "get_settings",
        lambda: Settings(
            llm_provider="qwen",
            llm_api_key="test-api-key",
            llm_model="qwen-max",
            llm_base_url="https://custom.example/v1",
        ),
    )
    service = llm_package.get_llm_service()
    assert service.model == "qwen-max"
    assert service.base_url == "https://custom.example/v1"
