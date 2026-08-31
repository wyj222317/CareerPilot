import pytest

from app.models.schemas import (
    CandidateProfile,
    JobRequirements,
    RecommendationNarrative,
    ResumeOptimizationResult,
)
from app.services.analysis import AnalysisError, MatchAnalysisService
from app.services.graph import build_analysis_graph
from app.services.llm.base import LLMService
from app.services.llm.mock import MockLLMService
from app.services.matching import MatchingService

RESUME = (
    "Jane Doe\n"
    "Skills: Python, React\n"
    "Senior Engineer - TechCorp (2020-2023)\n"
    "- Built REST APIs"
)
JOB = (
    "Backend Engineer\n"
    "Required: Python, Kubernetes\n"
    "Preferred: Docker\n"
    "3+ years of experience."
)


def run_graph(llm, matching=None, resume=RESUME, job=JOB):
    graph = build_analysis_graph(llm, matching)
    return graph.invoke({"raw_resume_text": resume, "raw_job_description": job})


class StubLLM(LLMService):
    """Deterministic stub with selectable failures and a call log."""

    def __init__(self, failures=()):
        self.failures = set(failures)
        self.calls = []

    def extract_candidate_profile(self, resume_text):
        self.calls.append("extract_candidate_profile")
        if "extract_candidate_profile" in self.failures:
            raise RuntimeError("profile boom")
        return CandidateProfile(
            name="Stub", skills=["Python", "React"], total_years_experience=5
        )

    def extract_job_requirements(self, job_description):
        self.calls.append("extract_job_requirements")
        if "extract_job_requirements" in self.failures:
            raise RuntimeError("requirements boom")
        return JobRequirements(
            title="Backend Engineer",
            required_skills=["Python", "Kubernetes"],
            preferred_skills=["Docker"],
            min_years_experience=3,
        )

    def generate_recommendation(self, profile, requirements, match):
        self.calls.append("generate_recommendation")
        if "generate_recommendation" in self.failures:
            raise RuntimeError("narrative boom")
        return RecommendationNarrative(
            summary="Stub summary.",
            strengths=["Stub strength"],
            recommendations=["Stub recommendation"],
        )

    def generate_resume_optimization(
        self, requirements, match, evidence_report, policy_actions, gap_warnings, relevant_evidence
    ):
        self.calls.append("generate_resume_optimization")
        if "generate_resume_optimization" in self.failures:
            raise RuntimeError("optimization boom")
        return ResumeOptimizationResult(
            actions=[],
            rewritten_bullets=[],
            gap_warnings=[],
            validation_issues=[],
            summary="Stub optimization.",
        )


class SpyMatchingService(MatchingService):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate_match(self, profile, requirements):
        self.calls += 1
        return super().calculate_match(profile, requirements)


def test_happy_path_with_mock_provider():
    state = run_graph(MockLLMService())
    assert state["candidate_profile"] is not None
    assert state["job_requirements"] is not None
    match = state["match_result"]
    assert match is not None
    assert len(match.score_breakdown) == 5
    assert state.get("errors", []) == []
    assert match.strengths  # mock narrative applied on top of deterministic match


def test_stub_flow_executes_nodes_in_order():
    llm = StubLLM()
    spy = SpyMatchingService()
    resume = (
        "Jane Doe\n"
        "Skills\n"
        "Python, React\n"
        "Experience\n"
        "Senior Engineer - TechCorp (2020-2023)\n"
        "- Built REST APIs"
    )
    job = (
        "Backend Engineer\n"
        "Required: Python, Kubernetes\n"
        "Preferred: Docker\n"
        "3+ years of experience."
    )
    run_graph(llm, spy, resume=resume, job=job)
    assert llm.calls == [
        "extract_candidate_profile",
        "extract_job_requirements",
        "generate_recommendation",
        "generate_resume_optimization",
    ]
    assert spy.calls == 1


def test_narrative_replaces_seeded_fields_but_not_scores():
    state = run_graph(StubLLM())
    match = state["match_result"]
    assert match.summary == "Stub summary."
    assert match.strengths == ["Stub strength"]
    assert match.recommendations == ["Stub recommendation"]
    assert len(match.score_breakdown) == 5
    assert match.overall_score >= 0


def test_resume_extraction_failure_aborts_before_matching():
    llm = StubLLM(failures={"extract_candidate_profile"})
    spy = SpyMatchingService()
    state = run_graph(llm, spy)
    assert state.get("candidate_profile") is None
    assert state.get("job_requirements") is not None  # second extraction still runs
    assert state.get("match_result") is None
    assert spy.calls == 0
    assert state["errors"] == ["parse_resume: profile boom"]


def test_job_extraction_failure_aborts_before_matching():
    llm = StubLLM(failures={"extract_job_requirements"})
    spy = SpyMatchingService()
    state = run_graph(llm, spy)
    assert state.get("candidate_profile") is not None
    assert state.get("job_requirements") is None
    assert state.get("match_result") is None
    assert spy.calls == 0
    assert state["errors"] == ["analyze_job_description: requirements boom"]


def test_recommendation_failure_degrades_gracefully():
    llm = StubLLM(failures={"generate_recommendation"})
    state = run_graph(llm)
    match = state["match_result"]
    assert match is not None  # deterministic result survives
    assert match.summary  # matcher's factual summary kept
    assert match.strengths  # matcher's seeded strengths kept
    assert match.missing_skills  # skill gaps still reported
    assert state["errors"] == ["generate_grounded_recommendation: narrative boom"]


def test_service_raises_analysis_error_when_extraction_fails():
    service = MatchAnalysisService(StubLLM(failures={"extract_candidate_profile"}))
    with pytest.raises(AnalysisError, match="parse_resume"):
        service.analyze(RESUME, JOB)


def test_service_returns_full_response_on_success():
    response = MatchAnalysisService(MockLLMService()).analyze(RESUME, JOB)
    assert response.candidate.name == "Jane Doe"
    assert response.job_requirements.required_skills
    assert len(response.match.score_breakdown) == 5
