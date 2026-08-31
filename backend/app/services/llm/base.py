from abc import ABC, abstractmethod

from app.models.schemas import (
    CandidateProfile,
    EvidenceReport,
    JobRequirements,
    MatchResult,
    RecommendationNarrative,
    ResumeOptimizationResult,
)


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot produce a valid structured output."""


class LLMService(ABC):
    """Provider-agnostic interface for LLM-backed text understanding.

    The LLM is responsible only for understanding unstructured text:
    parsing a resume, parsing a job description, and writing the narrative
    recommendation. Match scoring is deterministic and lives in
    MatchingService, not here.

    Implementations (mock today; OpenAI-compatible for OpenAI, DeepSeek and
    Qwen) plug in behind this interface so the rest of the app never depends
    on a specific vendor.
    """

    @abstractmethod
    def extract_candidate_profile(self, resume_text: str) -> CandidateProfile: ...

    @abstractmethod
    def extract_job_requirements(self, job_description: str) -> JobRequirements: ...

    @abstractmethod
    def generate_recommendation(
        self, profile: CandidateProfile, requirements: JobRequirements, match: MatchResult
    ) -> RecommendationNarrative: ...

    def generate_grounded_recommendation(
        self,
        profile: CandidateProfile,
        requirements: JobRequirements,
        match: MatchResult,
        evidence_report: EvidenceReport | None = None,
    ) -> RecommendationNarrative:
        """Generate a grounded recommendation using retrieved evidence.

        Default implementation delegates to generate_recommendation, ignoring
        evidence. Subclasses can override to use evidence in the prompt.
        """
        return self.generate_recommendation(profile, requirements, match)

    def generate_resume_optimization(
        self,
        requirements: JobRequirements,
        match: MatchResult,
        evidence_report: EvidenceReport,
        policy_actions: list,
        gap_warnings: list,
        relevant_evidence: list,
    ) -> ResumeOptimizationResult:
        """Generate resume optimization suggestions constrained by evidence.

        Default implementation returns an empty result so existing test doubles
        and simple providers do not need to implement optimization.
        """
        return ResumeOptimizationResult(
            actions=[],
            rewritten_bullets=[],
            gap_warnings=[],
            validation_issues=[],
            summary="Resume optimization not implemented for this provider.",
        )
