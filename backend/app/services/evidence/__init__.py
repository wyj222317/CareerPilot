"""Evidence grounding services for RAG-based traceability."""

from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import (
    EmbeddingService,
    MockEmbeddingService,
    OpenAICompatibleEmbeddingService,
    get_embedding_service,
)
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    "ResumeChunker",
    "EmbeddingService",
    "MockEmbeddingService",
    "OpenAICompatibleEmbeddingService",
    "get_embedding_service",
    "VectorStore",
    "InMemoryVectorStore",
    "EvidenceRetriever",
]
