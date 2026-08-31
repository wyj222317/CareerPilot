"""Tests for grounded recommendation generation."""

from app.models.schemas import (
    CandidateProfile,
    ClaimEvidence,
    EvidenceMatch,
    EvidenceReport,
    JobRequirements,
    MatchResult,
    ScoreComponent,
)
from app.services.llm.mock import MockLLMService


def test_mock_grounded_recommendation_with_evidence():
    llm = MockLLMService()
    profile = CandidateProfile(name="Test", skills=["Python"])
    requirements = JobRequirements(required_skills=["Python"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["python"],
        score_breakdown=[
            ScoreComponent(name="required_skills", weight=0.4, score=100, detail="test")
        ],
    )
    evidence_report = EvidenceReport(
        claims=[
            ClaimEvidence(
                claim="python",
                claim_type="required_skill",
                support_level="supported",
                evidence=[
                    EvidenceMatch(
                        evidence_id="exp-0-b0",
                        text="Built with Python",
                        section="experience",
                        score=0.8,
                    )
                ],
            )
        ],
        supported_count=1,
        partially_supported_count=0,
        unsupported_count=0,
    )

    narrative = llm.generate_grounded_recommendation(
        profile, requirements, match, evidence_report
    )
    # Should include evidence summary
    assert "1 supported" in narrative.summary
    assert "0 partial" in narrative.summary
    assert "0 unsupported" in narrative.summary


def test_mock_grounded_recommendation_without_evidence():
    llm = MockLLMService()
    profile = CandidateProfile(name="Test", skills=["Python"])
    requirements = JobRequirements(required_skills=["Python"])
    match = MatchResult(
        overall_score=100,
        summary="test",
        matched_skills=["python"],
        score_breakdown=[
            ScoreComponent(name="required_skills", weight=0.4, score=100, detail="test")
        ],
    )

    narrative = llm.generate_grounded_recommendation(
        profile, requirements, match, evidence_report=None
    )
    # Should fall back to regular recommendation
    assert "100/100" in narrative.summary


def test_grounded_recommendation_preserves_score():
    llm = MockLLMService()
    profile = CandidateProfile(name="Test", skills=["Python"])
    requirements = JobRequirements(required_skills=["Python"])
    match = MatchResult(
        overall_score=85,
        summary="test",
        matched_skills=["python"],
        score_breakdown=[
            ScoreComponent(name="required_skills", weight=0.4, score=100, detail="test")
        ],
    )
    evidence_report = EvidenceReport(
        claims=[],
        supported_count=0,
        partially_supported_count=0,
        unsupported_count=0,
    )

    narrative = llm.generate_grounded_recommendation(
        profile, requirements, match, evidence_report
    )
    # Score should be preserved in summary
    assert "85/100" in narrative.summary
