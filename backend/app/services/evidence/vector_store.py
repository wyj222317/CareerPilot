"""Vector store abstraction for evidence retrieval.

Provides:
- VectorStore ABC
- InMemoryVectorStore (numpy cosine similarity, exact and deterministic)
"""

from abc import ABC, abstractmethod

import numpy as np

from app.models.schemas import ResumeEvidence


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, evidence: list[ResumeEvidence], vectors: list[list[float]]) -> None:
        """Add evidence chunks with their embedding vectors."""
        ...

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[ResumeEvidence, float]]:
        """Search for the top_k most similar evidence chunks to the query."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored evidence and vectors."""
        ...

    @abstractmethod
    def all_evidence(self) -> list[ResumeEvidence]:
        """Return all stored evidence chunks (for lexical fallback)."""
        ...


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using numpy for cosine similarity.

    Exact, deterministic, and lightweight for small corpora (tens of chunks).
    """

    def __init__(self):
        self._evidence: list[ResumeEvidence] = []
        self._vectors: list[np.ndarray] = []

    def add(self, evidence: list[ResumeEvidence], vectors: list[list[float]]) -> None:
        if len(evidence) != len(vectors):
            raise ValueError("evidence and vectors must have the same length")
        self._evidence.extend(evidence)
        self._vectors.extend(np.array(v, dtype=np.float32) for v in vectors)

    def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[ResumeEvidence, float]]:
        if not self._evidence:
            return []

        query = np.array(query_vector, dtype=np.float32)
        # L2 normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Stack all vectors into a matrix
        matrix = np.stack(self._vectors)
        # Cosine similarity = dot product (since vectors are already normalized)
        similarities = matrix @ query

        # Get top_k indices
        k = min(top_k, len(self._evidence))
        top_indices = np.argsort(similarities, kind="stable")[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((self._evidence[idx], float(similarities[idx])))
        return results

    def clear(self) -> None:
        self._evidence.clear()
        self._vectors.clear()

    def all_evidence(self) -> list[ResumeEvidence]:
        return list(self._evidence)
