"""Deterministic policy layer for evidence-constrained resume optimization.

The policy decides what the optimizer is allowed to do for each JD claim based
on its evidence support level. It never invents facts; it only selects which
existing evidence may be rewritten, surfaced, or reordered, and which missing
requirements must become gap warnings.
"""

from __future__ import annotations

from app.models.schemas import (
    ClaimEvidence,
    EvidenceReport,
    GapWarning,
    JobRequirements,
    OptimizationAction,
    OptimizationActionType,
)
from app.services.skill_normalization import SkillNormalizer


class OptimizationPolicy:
    """Map evidence support levels to allowed optimization actions."""

    def __init__(self, normalizer: SkillNormalizer | None = None) -> None:
        self._normalizer = normalizer or SkillNormalizer()

    def decide_action_type(self, support_level: str) -> OptimizationActionType:
        """Return the action type allowed for a given support level."""
        if support_level == "supported":
            return "rewrite"
        if support_level == "partially_supported":
            return "rewrite"
        return "gap_warning"

    def build_guidance(self, claim_evidence: ClaimEvidence) -> str:
        """Return a human-readable constraint string for the LLM."""
        if claim_evidence.support_level == "supported":
            return (
                "This requirement is fully supported by evidence. You may rewrite, "
                "reorder, or surface the cited evidence to align terminology with "
                "the job description. Do not add new facts."
            )
        if claim_evidence.support_level == "partially_supported":
            return (
                "This requirement is only partially supported. Rewrite only the "
                "supported portion. Do not imply the unsupported portion is true."
            )
        return (
            "This requirement is unsupported by the resume. Produce a gap_warning "
            "instead of rewritten content. Do not invent experience or skills."
        )

    def select_evidence_ids(self, claim_evidence: ClaimEvidence) -> list[str]:
        """Return the evidence IDs that may be used for this claim."""
        return [e.evidence_id for e in claim_evidence.evidence]

    def target_section(self, claim_evidence: ClaimEvidence) -> str:
        """Suggest a resume section to optimize based on the retrieved evidence."""
        sections = {e.section for e in claim_evidence.evidence}
        if "experience" in sections:
            return "experience"
        if "summary" in sections:
            return "summary"
        if "skills" in sections:
            return "skills"
        if "education" in sections:
            return "education"
        return "summary"

    def _unsupported_skills(self, claim: ClaimEvidence) -> list[str]:
        """Return the atomic skills in *claim* that are not evidenced."""
        group = self._normalizer.parse_claim(claim.claim)
        if group is None:
            return []

        found_skills: set[str] = set()
        for ev in claim.evidence:
            found_skills.update(self._normalizer.extract_canonical_skills(ev.text))

        return sorted(set(group.skills) - found_skills)

    def generate_actions(
        self, requirements: JobRequirements, evidence_report: EvidenceReport
    ) -> tuple[list[OptimizationAction], list[GapWarning]]:
        """Generate initial policy actions and gap warnings from evidence report.

        The LLM optimizer uses these as guardrails; final output is still
        validated by OptimizationValidator.
        """
        actions: list[OptimizationAction] = []
        warnings: list[GapWarning] = []

        for claim in evidence_report.claims:
            action_type = self.decide_action_type(claim.support_level)
            evidence_ids = self.select_evidence_ids(claim)

            if action_type == "gap_warning":
                warnings.append(
                    GapWarning(
                        requirement=claim.claim,
                        rationale=(
                            f"No resume evidence supports this {claim.claim_type} "
                            "requirement. Add real experience or training before "
                            "claiming it on a resume."
                        ),
                    )
                )
                continue

            # For partially supported compound claims, also warn about the
            # specific unsupported atomic skills.
            if claim.support_level == "partially_supported":
                for skill in self._unsupported_skills(claim):
                    warnings.append(
                        GapWarning(
                            requirement=skill,
                            rationale=(
                                f"'{skill}' is required by '{claim.claim}' but is "
                                "not evidenced in the resume. Do not claim it "
                                "without real experience."
                            ),
                        )
                    )

            actions.append(
                OptimizationAction(
                    action_type=action_type,
                    target_section=self.target_section(claim),
                    original_text=claim.claim,
                    optimized_text=None,
                    rationale=self.build_guidance(claim),
                    target_requirements=[claim.claim],
                    evidence_ids=evidence_ids,
                    confidence=1.0 if claim.support_level == "supported" else 0.6,
                )
            )

        return actions, warnings
