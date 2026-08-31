"""Embedding service abstraction for evidence retrieval.

Provides:
- EmbeddingService ABC
- MockEmbeddingService (deterministic hash-based vectors for tests)
- OpenAICompatibleEmbeddingService (real provider via OpenAI SDK)
- get_embedding_service() factory
"""

from abc import ABC, abstractmethod
from functools import lru_cache

import zlib

import numpy as np

from app.core.config import get_settings


class EmbeddingService(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


class MockEmbeddingService(EmbeddingService):
    """Deterministic mock embedding service using token hashing.

    Produces vectors by hashing tokens into fixed dimensions, summing, and
    L2-normalizing. Same text always produces the same vector; semantically
    overlapping texts share token-hash dimensions.
    """

    def __init__(self, dimension: int = 256):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = np.zeros(self._dimension, dtype=np.float32)
            tokens = text.lower().split()
            for token in tokens:
                # Hash token into a dimension index
                idx = zlib.crc32(token.encode()) % self._dimension
                vec[idx] += 1.0
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors


class OpenAICompatibleEmbeddingService(EmbeddingService):
    """OpenAI-compatible embedding service using the openai SDK."""

    def __init__(
        self,
        client,
        model: str,
        dimension: int = 1536,
    ):
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Factory that returns the configured embedding service."""
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "mock":
        return MockEmbeddingService()

    if provider == "openai":
        from openai import OpenAI

        api_key = settings.embedding_api_key or settings.llm_api_key
        if not api_key:
            raise ValueError(
                "EMBEDDING_API_KEY or LLM_API_KEY must be set for provider=openai"
            )
        base_url = settings.embedding_base_url or "https://api.openai.com/v1"
        model = settings.embedding_model or "text-embedding-3-small"
        client = OpenAI(api_key=api_key, base_url=base_url)
        return OpenAICompatibleEmbeddingService(client=client, model=model)

    raise ValueError(f"Unknown embedding provider: {provider}")
