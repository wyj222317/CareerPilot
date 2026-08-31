"""Evidence-constrained resume optimizer.

The optimizer improves resume presentation for JD alignment while staying
within the factual boundary defined by retrieved resume evidence.
"""

from __future__ import annotations

from app.models.schemas import (
    EvidenceReport,
    JobRequirements,
    MatchResult,
    ResumeEvidence,
    ResumeOptimizationResult,
)
from app.services.llm.base import LLMService
from app.services.optimization.policy import OptimizationPolicy


class ResumeOptimizer:
    """Generate resume optimization suggestions constrained by evidence."""

    def __init__(
        self,
        llm_service: LLMService,
        policy: OptimizationPolicy | None = None,
    ) -> None:
        self._llm = llm_service
        self._policy = policy or OptimizationPolicy()

    def optimize(
        self,
        requirements: JobRequirements,
        match: MatchResult,
        evidence_report: EvidenceReport,
        resume_evidence: list[ResumeEvidence],
    ) -> ResumeOptimizationResult:
        """Produce an evidence-constrained optimization result.

        Optimization is best-effort: failures are caught and returned with
        validation issues, but they never modify the deterministic match result.
        """
        policy_actions, gap_warnings = self._policy.generate_actions(
            requirements, evidence_report
        )

        try:
            return self._llm.generate_resume_optimization(
                requirements=requirements,
                match=match,
                evidence_report=evidence_report,
                policy_actions=policy_actions,
                gap_warnings=gap_warnings,
                relevant_evidence=resume_evidence,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return ResumeOptimizationResult(
                actions=[],
                rewritten_bullets=[],
                gap_warnings=gap_warnings,
                validation_issues=[
                    {
                        "field": "llm",
                        "message": f"Optimization generation failed: {exc}",
                    }
                ],
                summary="Resume optimization could not be generated.",
            )
