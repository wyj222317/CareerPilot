"""Adversarial tests for evidence-constrained resume optimization."""

from app.models.schemas import (
    EvidenceMatch,
    EvidenceReport,
    JobRequirements,
    MatchResult,
    OptimizationAction,
    ResumeEvidence,
    ResumeOptimizationResult,
)
from app.services.evidence.embeddings import MockEmbeddingService
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore
from app.services.llm.mock import MockLLMService
from app.services.optimization.optimizer import ResumeOptimizer
from app.services.optimization.policy import OptimizationPolicy
from app.services.optimization.validator import OptimizationValidator


def _make_evidence(text: str, evidence_id: str = "exp-0") -> ResumeEvidence:
    return ResumeEvidence(
        id=evidence_id,
        text=text,
        section="experience",
        source="Experience",
    )


def _build_report(claim_text: str, support_level: str, evidence_texts: list[str]) -> EvidenceReport:
    evidence = [
        EvidenceMatch(
            evidence_id=f"exp-{i}",
            text=text,
            section="experience",
            score=1.0,
        )
        for i, text in enumerate(evidence_texts)
    ]
    return EvidenceReport(
        claims=[
            {
                "claim": claim_text,
                "claim_type": "required_skill",
                "support_level": support_level,
                "evidence": evidence,
            }
        ]
    )


def test_aws_unsupported_becomes_gap_warning_not_bullet():
    """Critical case 1: unsupported AWS must not be fabricated."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = EvidenceReport(
        claims=[
            {
                "claim": "FastAPI and AWS",
                "claim_type": "required_skill",
                "support_level": "partially_supported",
                "evidence": [
                    EvidenceMatch(
                        evidence_id="exp-0",
                        text="Built APIs using FastAPI.",
                        section="experience",
                        score=1.0,
                    )
                ],
            }
        ]
    )
    requirements = JobRequirements(required_skills=["FastAPI and AWS"])
    match = MatchResult(
        overall_score=50,
        summary="test",
        matched_skills=["fastapi"],
        missing_skills=[],
    )

    optimizer = ResumeOptimizer(MockLLMService())
    result = optimizer.optimize(requirements, match, report, evidence)

    # AWS should be a gap warning, not claimed in a rewrite.
    assert any("aws" in w.requirement.lower() for w in result.gap_warnings)
    rewritten_texts = [b.rewritten for b in result.rewritten_bullets]
    assert all("aws" not in text.lower() for text in rewritten_texts)


def test_numbers_in_optimized_text_must_appear_in_evidence():
    """Critical case 2: 20% allowed, 40% forbidden."""
    evidence = [_make_evidence("Improved latency by 20%.", "exp-0")]
    report = _build_report("Improved latency by 20%", "supported", ["Improved latency by 20%."])
    requirements = JobRequirements(required_skills=["latency optimization"])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["latency optimization"])

    optimizer = ResumeOptimizer(MockLLMService())
    result = optimizer.optimize(requirements, match, report, evidence)

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    for action in validated.actions:
        if action.optimized_text:
            assert "40%" not in action.optimized_text
            assert "20%" in action.optimized_text


def test_no_metric_in_evidence_means_no_metric_in_rewrite():
    """Critical case 3: cannot invent 30% performance improvement."""
    evidence = [_make_evidence("Improved backend performance.", "exp-0")]
    report = _build_report("performance", "supported", ["Improved backend performance."])
    requirements = JobRequirements(required_skills=["performance"])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["performance"])

    optimizer = ResumeOptimizer(MockLLMService())
    result = optimizer.optimize(requirements, match, report, evidence)

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    for action in validated.actions:
        if action.optimized_text:
            assert "30%" not in action.optimized_text


def test_cannot_invent_employer_or_deployment_platform():
    """Critical case 4: Docker evidence cannot become AWS deployment claim."""
    evidence = [_make_evidence("Used Docker for containerization.", "exp-0")]
    report = _build_report("Docker", "supported", ["Used Docker for containerization."])
    requirements = JobRequirements(required_skills=["AWS deployment"])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["docker"])

    optimizer = ResumeOptimizer(MockLLMService())
    result = optimizer.optimize(requirements, match, report, evidence)

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    for action in validated.actions:
        if action.optimized_text:
            assert "AWS" not in action.optimized_text
            assert "deployed" not in action.optimized_text.lower()


def test_cannot_rename_skill_to_unsupported_alternative():
    """Critical case 5: LangChain evidence cannot be rewritten as LangGraph."""
    evidence = [_make_evidence("Worked with LangChain.", "exp-0")]
    report = _build_report("LangGraph or LangChain", "supported", ["Worked with LangChain."])
    requirements = JobRequirements(required_skills=["LangGraph or LangChain"])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["langchain"])

    optimizer = ResumeOptimizer(MockLLMService())
    result = optimizer.optimize(requirements, match, report, evidence)

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    for action in validated.actions:
        if action.optimized_text:
            assert "LangGraph" not in action.optimized_text


def test_rewrite_action_must_reference_valid_evidence_ids():
    """Critical case 6: every rewrite must cite real evidence."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    invalid_result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="rewrite",
                target_section="experience",
                original_text="Built APIs using FastAPI.",
                optimized_text="Built APIs using FastAPI.",
                rationale="test",
                evidence_ids=["nonexistent-id"],
            )
        ],
        summary="test",
    )
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    requirements = JobRequirements(required_skills=["FastAPI"])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])

    validator = OptimizationValidator(evidence)
    validated = validator.validate(invalid_result, report, match)

    assert len(validated.actions) == 0
    assert any("nonexistent-id" in i.message for i in validated.validation_issues)


def test_invalid_evidence_id_is_caught_by_validator():
    """Critical case 7: invalid evidence ID fails validation."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="rewrite",
                target_section="experience",
                optimized_text="Built APIs using FastAPI.",
                rationale="test",
                evidence_ids=["exp-0", "missing-id"],
            )
        ],
        summary="test",
    )
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert any("missing-id" in i.message for i in validated.validation_issues)


def test_optimization_does_not_modify_match_score():
    """Critical case 8: MatchResult score fields are untouched."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    requirements = JobRequirements(required_skills=["FastAPI"])
    match = MatchResult(overall_score=75, summary="test", matched_skills=["fastapi"])

    optimizer = ResumeOptimizer(MockLLMService())
    optimizer.optimize(requirements, match, report, evidence)

    assert match.overall_score == 75
    assert match.score_breakdown == []


def test_optimization_failure_preserves_phase_3_result():
    """Critical case 9: a failing optimization node must not destroy prior state."""
    from app.models.schemas import (
        CandidateProfile,
        JobRequirements,
        RecommendationNarrative,
        ResumeOptimizationResult,
    )
    from app.services.graph import build_analysis_graph
    from app.services.llm.base import LLMService

    class FailingOptimizationLLM(LLMService):
        """LLM that succeeds through evidence retrieval but fails optimization."""

        def extract_candidate_profile(self, resume_text: str) -> CandidateProfile:
            return CandidateProfile(
                name="Jane Doe",
                skills=["Python", "React"],
                total_years_experience=5,
            )

        def extract_job_requirements(self, job_description: str) -> JobRequirements:
            return JobRequirements(
                title="Backend Engineer",
                required_skills=["Python", "Kubernetes"],
                preferred_skills=["Docker"],
                min_years_experience=3,
            )

        def generate_recommendation(
            self, profile, requirements, match
        ) -> RecommendationNarrative:
            return RecommendationNarrative(summary="ok")

        def generate_resume_optimization(
            self, requirements, match, evidence_report, policy_actions, gap_warnings, relevant_evidence
        ) -> ResumeOptimizationResult:
            raise RuntimeError("optimization boom")

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

    graph = build_analysis_graph(FailingOptimizationLLM())
    state = graph.invoke(
        {
            "raw_resume_text": resume,
            "raw_job_description": job,
        }
    )

    assert state["match_result"] is not None
    assert state["evidence_report"] is not None
    # Optimization may be None or empty, but Phase 3 results remain.
    optimization = state.get("resume_optimization")
    assert optimization is None or not optimization.actions


def test_optimization_is_deterministic():
    """Critical case 10: same inputs produce same mock output."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    requirements = JobRequirements(required_skills=["FastAPI"])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])

    optimizer = ResumeOptimizer(MockLLMService())
    result_1 = optimizer.optimize(requirements, match, report, evidence)
    result_2 = optimizer.optimize(requirements, match, report, evidence)

    assert result_1.model_dump() == result_2.model_dump()


def test_unsupported_requirement_cannot_produce_rewrite():
    """Unsupported requirement must only produce gap warnings."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = EvidenceReport(
        claims=[
            {
                "claim": "Kubernetes",
                "claim_type": "required_skill",
                "support_level": "unsupported",
                "evidence": [],
            }
        ]
    )
    requirements = JobRequirements(required_skills=["Kubernetes"])
    match = MatchResult(overall_score=50, summary="test", matched_skills=[])

    optimizer = ResumeOptimizer(MockLLMService())
    result = optimizer.optimize(requirements, match, report, evidence)

    assert any("Kubernetes" in w.requirement for w in result.gap_warnings)
    assert not any(a.action_type == "rewrite" for a in result.actions)


def test_surface_existing_evidence_without_optimized_text_fails_validation():
    """A surface_existing_evidence action without optimized_text is structurally invalid."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])
    result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="surface_existing_evidence",
                target_section="experience",
                original_text="Built APIs using FastAPI.",
                optimized_text=None,
                rationale="Surface existing evidence.",
                target_requirements=["FastAPI"],
                evidence_ids=["exp-0"],
            )
        ],
        summary="test",
    )

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert len(validated.actions) == 0
    assert any(
        "optimized_text" in i.field and "surface_existing_evidence" in i.message
        for i in validated.validation_issues
    )


def test_surface_existing_evidence_with_optimized_text_and_valid_evidence_passes():
    """A complete surface_existing_evidence action passes structural validation."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])
    result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="surface_existing_evidence",
                target_section="experience",
                original_text="Built APIs using FastAPI.",
                optimized_text="Built APIs using FastAPI.",
                rationale="Surface existing evidence.",
                target_requirements=["FastAPI"],
                evidence_ids=["exp-0"],
            )
        ],
        summary="test",
    )

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert len(validated.actions) == 1
    assert len(validated.validation_issues) == 0


def test_rewrite_without_optimized_text_fails_validation():
    """A rewrite action must provide optimized_text."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])
    result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="rewrite",
                target_section="experience",
                original_text="Built APIs using FastAPI.",
                optimized_text=None,
                rationale="Rewrite.",
                target_requirements=["FastAPI"],
                evidence_ids=["exp-0"],
            )
        ],
        summary="test",
    )

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert len(validated.actions) == 0
    assert any("optimized_text" in i.field for i in validated.validation_issues)


def test_rewrite_without_original_text_fails_validation():
    """A rewrite action must identify the existing bullet being rewritten."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])
    result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="rewrite",
                target_section="experience",
                original_text=None,
                optimized_text="Built APIs using FastAPI.",
                rationale="Rewrite.",
                target_requirements=["FastAPI"],
                evidence_ids=["exp-0"],
            )
        ],
        summary="test",
    )

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert len(validated.actions) == 0
    assert any("original_text" in i.field for i in validated.validation_issues)


def test_unsupported_skill_cannot_be_surfaced():
    """surface_existing_evidence must not introduce skills absent from cited evidence."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])
    result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="surface_existing_evidence",
                target_section="experience",
                original_text="Built APIs using FastAPI.",
                optimized_text="Built APIs using FastAPI and Kubernetes.",
                rationale="Surface with unsupported addition.",
                target_requirements=["FastAPI"],
                evidence_ids=["exp-0"],
            )
        ],
        summary="test",
    )

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert len(validated.actions) == 0
    assert any("kubernetes" in i.message.lower() for i in validated.validation_issues)


def test_gap_warning_action_is_rejected():
    """gap_warning entries must live in gap_warnings, not actions."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    match = MatchResult(overall_score=100, summary="test", matched_skills=["fastapi"])
    result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="gap_warning",
                target_section="experience",
                rationale="Missing requirement.",
                target_requirements=["FastAPI"],
            )
        ],
        summary="test",
    )

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert len(validated.actions) == 0
    assert any("gap_warning" in i.message for i in validated.validation_issues)


def test_real_mock_optimization_has_zero_structural_validation_issues():
    """The deterministic mock LLM must produce structurally valid optimization output."""
    evidence = [_make_evidence("Built APIs using FastAPI.", "exp-0")]
    report = _build_report("FastAPI", "supported", ["Built APIs using FastAPI."])
    requirements = JobRequirements(required_skills=["FastAPI"])
    match = MatchResult(
        overall_score=100, summary="test", matched_skills=["fastapi"]
    )

    optimizer = ResumeOptimizer(MockLLMService())
    result = optimizer.optimize(requirements, match, report, evidence)

    validator = OptimizationValidator(evidence)
    validated = validator.validate(result, report, match)

    assert len(validated.actions) == len(result.actions)
    assert len(validated.validation_issues) == 0
