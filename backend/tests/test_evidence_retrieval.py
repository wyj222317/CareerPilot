"""Tests for evidence retrieval service support classification."""

from app.models.schemas import (
    JobRequirements,
    MatchResult,
    ResumeEvidence,
    ScoreComponent,
)
from app.services.evidence.embeddings import MockEmbeddingService
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore


def _build_store_with_evidence(embedder, evidence_list):
    store = InMemoryVectorStore()
    texts = [e.text for e in evidence_list]
    vectors = embedder.embed(texts)
    store.add(evidence_list, vectors)
    return store


def test_supported_fastapi_explicit_mention():
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder)

    evidence = [
        ResumeEvidence(
            id="exp-0-b0",
            text="Built REST APIs with FastAPI and Pydantic",
            section="experience",
            source="Experience",
        )
    ]
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["FastAPI"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["fastapi"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = report.claims[0]
    assert claim.claim == "FastAPI"
    assert claim.support_level == "supported"


def test_supported_docker_and_aws_across_one_chunk():
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder)

    evidence = [
        ResumeEvidence(
            id="exp-0-b0",
            text="Deployed containerized services with Docker and Kubernetes on AWS",
            section="experience",
            source="Experience",
        )
    ]
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["Experience deploying services with Docker and AWS"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["docker", "aws"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = next(c for c in report.claims if c.claim_type == "required_skill")
    assert claim.support_level == "supported"


def test_supported_fastapi_and_pydantic_across_separate_chunks():
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder)

    evidence = [
        ResumeEvidence(
            id="exp-0-b0",
            text="Built APIs using FastAPI",
            section="experience",
            source="Experience",
        ),
        ResumeEvidence(
            id="exp-1-b0",
            text="Used Pydantic models for validation",
            section="experience",
            source="Experience",
        ),
    ]
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["FastAPI and Pydantic"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["fastapi", "pydantic"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = report.claims[0]
    assert claim.support_level == "supported"


def test_partially_supported_and_requirement():
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder)

    evidence = [
        ResumeEvidence(
            id="exp-0-b0",
            text="Built APIs using FastAPI",
            section="experience",
            source="Experience",
        )
    ]
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["FastAPI and Pydantic"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["fastapi"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = report.claims[0]
    assert claim.support_level == "partially_supported"


def test_or_with_one_alternative_supported():
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder)

    evidence = [
        ResumeEvidence(
            id="exp-0-b0",
            text="Experience with LangGraph",
            section="experience",
            source="Experience",
        )
    ]
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["LangGraph or LangChain"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["langgraph"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = report.claims[0]
    assert claim.support_level == "supported"


def test_unsupported_no_explicit_technical_evidence():
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder)

    evidence = [
        ResumeEvidence(
            id="exp-0-b0",
            text="Worked on backend services",
            section="experience",
            source="Experience",
        )
    ]
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["FastAPI"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["fastapi"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = report.claims[0]
    assert claim.support_level == "unsupported"


def test_retrieval_ranking_does_not_modify_support_semantics():
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder, top_k=2)

    # First chunk is irrelevant but might rank high due to embedding noise;
    # second chunk contains the actual skill.
    evidence = [
        ResumeEvidence(
            id="exp-0-b0",
            text="Team player with strong communication skills",
            section="experience",
            source="Experience",
        ),
        ResumeEvidence(
            id="exp-1-b0",
            text="Built services with FastAPI",
            section="experience",
            source="Experience",
        ),
    ]
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["FastAPI"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["fastapi"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = report.claims[0]
    # Should be supported because FastAPI is explicitly evidenced in retrieved chunks
    assert claim.support_level == "supported"


def test_deterministic_score_before_after_evidence_unchanged():
    from app.services.matching import MatchingService

    requirements = JobRequirements(required_skills=["FastAPI"])
    from app.models.schemas import CandidateProfile

    profile = CandidateProfile(skills=["FastAPI"])
    match_before = MatchingService().calculate_match(profile, requirements)

    # Evidence retrieval does not touch MatchResult score fields
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder)
    store = InMemoryVectorStore()
    report = retriever.retrieve(requirements, match_before, store)

    assert match_before.overall_score == 100
    assert report is not None


def test_lexical_fallback_finds_short_skill_missed_by_vectors():
    """Mock embeddings can miss short skills; lexical fallback must include them."""
    embedder = MockEmbeddingService(dimension=128)
    retriever = EvidenceRetriever(embedder=embedder, top_k=2)

    # Create many chunks so the skills chunk may fall outside top_k.
    evidence = [
        ResumeEvidence(
            id=f"exp-{i}",
            text=f"Worked on project {i} with various technologies",
            section="experience",
            source="Experience",
        )
        for i in range(5)
    ]
    evidence.append(
        ResumeEvidence(
            id="skills",
            text="Python, FastAPI, Pydantic",
            section="skills",
            source="Skills",
        )
    )
    store = _build_store_with_evidence(embedder, evidence)

    requirements = JobRequirements(required_skills=["Pydantic"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["pydantic"],
    )

    report = retriever.retrieve(requirements, match, store)
    claim = report.claims[0]
    assert claim.support_level == "supported"
    assert any(e.evidence_id == "skills" for e in claim.evidence)
