"""Tests for vector store abstraction."""

import numpy as np

from app.models.schemas import ResumeEvidence
from app.services.evidence.vector_store import InMemoryVectorStore


def test_vector_store_add_and_search():
    store = InMemoryVectorStore()
    evidence = [
        ResumeEvidence(id="e1", text="Python FastAPI", section="skills", source="Skills"),
        ResumeEvidence(id="e2", text="Java Spring", section="skills", source="Skills"),
    ]
    vectors = [
        [1.0, 0.0, 0.0],  # e1
        [0.0, 1.0, 0.0],  # e2
    ]
    store.add(evidence, vectors)
    # Search for Python-like query
    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "e1"
    assert results[0][1] > 0.99  # High similarity


def test_vector_store_search_returns_top_k():
    store = InMemoryVectorStore()
    evidence = [
        ResumeEvidence(id=f"e{i}", text=f"text{i}", section="skills", source="Skills")
        for i in range(5)
    ]
    vectors = [[float(i), 0.0, 0.0] for i in range(5)]
    store.add(evidence, vectors)
    results = store.search([4.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    # Should return e4 and e3 (highest similarity)
    assert results[0][0].id == "e4"
    assert results[1][0].id == "e3"


def test_vector_store_empty_search():
    store = InMemoryVectorStore()
    results = store.search([1.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_vector_store_clear():
    store = InMemoryVectorStore()
    evidence = [ResumeEvidence(id="e1", text="test", section="skills", source="Skills")]
    vectors = [[1.0, 0.0, 0.0]]
    store.add(evidence, vectors)
    store.clear()
    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert results == []


def test_vector_store_cosine_similarity_ordering():
    store = InMemoryVectorStore()
    # Create vectors with known cosine similarities
    evidence = [
        ResumeEvidence(id="e1", text="close", section="skills", source="Skills"),
        ResumeEvidence(id="e2", text="far", section="skills", source="Skills"),
    ]
    # e1 is close to query, e2 is far
    vectors = [
        [1.0, 0.0, 0.0],  # e1: cosine=1.0 with query
        [0.0, 0.0, 1.0],  # e2: cosine=0.0 with query
    ]
    store.add(evidence, vectors)
    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert results[0][0].id == "e1"
    assert results[1][0].id == "e2"
    assert results[0][1] > results[1][1]


def test_vector_store_length_mismatch_raises():
    store = InMemoryVectorStore()
    evidence = [ResumeEvidence(id="e1", text="test", section="skills", source="Skills")]
    vectors = [[1.0, 0.0], [0.0, 1.0]]  # 2 vectors for 1 evidence
    try:
        store.add(evidence, vectors)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "same length" in str(e)


def test_vector_store_all_evidence():
    store = InMemoryVectorStore()
    evidence = [
        ResumeEvidence(id="e1", text="Python", section="skills", source="Skills"),
        ResumeEvidence(id="e2", text="Java", section="skills", source="Skills"),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    store.add(evidence, vectors)
    all_ev = store.all_evidence()
    assert len(all_ev) == 2
    assert {e.id for e in all_ev} == {"e1", "e2"}
