"""Tests for embedding service abstraction."""

import numpy as np

from app.services.evidence.embeddings import MockEmbeddingService


def test_mock_embedding_deterministic():
    embedder = MockEmbeddingService(dimension=128)
    texts = ["Python FastAPI Docker", "Python FastAPI Docker"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 2
    # Same text should produce identical vectors
    assert vectors[0] == vectors[1]


def test_mock_embedding_different_texts_different_vectors():
    embedder = MockEmbeddingService(dimension=128)
    texts = ["Python FastAPI", "Java Spring"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 2
    # Different texts should produce different vectors
    assert vectors[0] != vectors[1]


def test_mock_embedding_dimension():
    embedder = MockEmbeddingService(dimension=256)
    assert embedder.dimension == 256
    vectors = embedder.embed(["test text"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 256


def test_mock_embedding_normalized():
    embedder = MockEmbeddingService(dimension=128)
    vectors = embedder.embed(["Python FastAPI Docker"])
    vec = np.array(vectors[0])
    norm = np.linalg.norm(vec)
    # Should be L2 normalized (approximately 1.0)
    assert abs(norm - 1.0) < 1e-5


def test_mock_embedding_semantic_overlap():
    embedder = MockEmbeddingService(dimension=128)
    # Texts with overlapping tokens should have higher similarity
    vec1 = np.array(embedder.embed(["Python FastAPI Docker"])[0])
    vec2 = np.array(embedder.embed(["Python FastAPI Kubernetes"])[0])
    vec3 = np.array(embedder.embed(["Java Spring Hibernate"])[0])
    # Similarity = dot product (since normalized)
    sim_12 = np.dot(vec1, vec2)
    sim_13 = np.dot(vec1, vec3)
    # Overlapping texts should have higher similarity
    assert sim_12 > sim_13


def test_mock_embedding_empty_text():
    embedder = MockEmbeddingService(dimension=128)
    vectors = embedder.embed([""])
    assert len(vectors) == 1
    vec = np.array(vectors[0])
    # Empty text should produce zero vector
    assert np.allclose(vec, 0.0)
