"""Post-generation validation for resume optimization output.

Every optimization action must be grounded in cited evidence. The validator
rejects actions that invent facts, numbers, employers, skills, or unsupported
claims. Invalid actions are dropped and recorded as validation issues so the
rest of the optimization result is still usable.
"""

from __future__ import annotations

import re

from app.models.schemas import (
    ClaimEvidence,
    EvidenceReport,
    MatchResult,
    OptimizationAction,
    ResumeEvidence,
    ResumeOptimizationResult,
    ValidationIssue,
)
from app.services.factual_concepts import extract_factual_concepts
from app.services.llm.mock import find_skills

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class OptimizationValidator:
    """Validate that optimized actions stay within the factual boundary."""

    def __init__(self, evidence: list[ResumeEvidence]) -> None:
        self._evidence_by_id = {e.id: e for e in evidence}

    def validate(
        self,
        result: ResumeOptimizationResult,
        evidence_report: EvidenceReport,
        match: MatchResult,
    ) -> ResumeOptimizationResult:
        """Drop invalid actions and collect validation issues."""
        valid_actions: list[OptimizationAction] = []
        issues: list[ValidationIssue] = list(result.validation_issues)

        support_by_claim = {c.claim: c for c in evidence_report.claims}

        for action in result.actions:
            action_issues = self._validate_action(action, support_by_claim)
            if action_issues:
                issues.extend(action_issues)
            else:
                valid_actions.append(action)

        # Score immutability is structural: MatchResult is never passed to the
        # optimizer, so we only sanity-check the provided match object.
        if match.overall_score < 0 or match.overall_score > 100:
            issues.append(
                ValidationIssue(
                    field="match.overall_score",
                    message=f"Invalid overall score: {match.overall_score}",
                )
            )

        return ResumeOptimizationResult(
            actions=valid_actions,
            rewritten_bullets=result.rewritten_bullets,
            gap_warnings=result.gap_warnings,
            validation_issues=issues,
            summary=result.summary,
        )

    def _cited_texts(self, action: OptimizationAction) -> list[str]:
        return [
            self._evidence_by_id[eid].text
            for eid in action.evidence_ids
            if eid in self._evidence_by_id
        ]

    def _validate_action(
        self,
        action: OptimizationAction,
        support_by_claim: dict[str, ClaimEvidence],
    ) -> list[ValidationIssue]:
        """Return validation issues for a single action."""
        issues: list[ValidationIssue] = []

        # A. gap_warning belongs in the gap_warnings array, not actions.
        if action.action_type == "gap_warning":
            issues.append(
                ValidationIssue(
                    field="action_type",
                    message="gap_warning must be emitted in gap_warnings, not actions.",
                )
            )

        # B. Evidence IDs must exist.
        for evidence_id in action.evidence_ids:
            if evidence_id not in self._evidence_by_id:
                issues.append(
                    ValidationIssue(
                        field="evidence_ids",
                        message=f"Evidence ID '{evidence_id}' does not exist.",
                    )
                )

        # C. Structurally complete resume-facing suggestions.
        if action.action_type in ("rewrite", "surface_existing_evidence"):
            if not action.evidence_ids:
                issues.append(
                    ValidationIssue(
                        field="evidence_ids",
                        message=(
                            f"{action.action_type} action for "
                            f"'{action.target_section}' must cite evidence."
                        ),
                    )
                )
            if not action.optimized_text:
                issues.append(
                    ValidationIssue(
                        field="optimized_text",
                        message=(
                            f"{action.action_type} action must provide "
                            "optimized_text."
                        ),
                    )
                )

        if action.action_type == "rewrite" and not action.original_text:
            issues.append(
                ValidationIssue(
                    field="original_text",
                    message="rewrite action must provide original_text.",
                )
            )

        if action.action_type == "reorder":
            if not action.original_text:
                issues.append(
                    ValidationIssue(
                        field="original_text",
                        message="reorder action must identify the existing content.",
                    )
                )
            if not action.evidence_ids:
                issues.append(
                    ValidationIssue(
                        field="evidence_ids",
                        message="reorder action must cite evidence for the existing content.",
                    )
                )

        # D. Unsupported requirements cannot produce rewrite/surface/reorder actions.
        if action.action_type in ("rewrite", "surface_existing_evidence", "reorder"):
            for req in action.target_requirements:
                claim = support_by_claim.get(req)
                if claim is not None and claim.support_level == "unsupported":
                    issues.append(
                        ValidationIssue(
                            field="target_requirements",
                            message=(
                                f"Cannot rewrite unsupported requirement: '{req}'."
                            ),
                        )
                    )

        if not action.optimized_text:
            return issues

        cited_texts = self._cited_texts(action)
        cited_text = " ".join(cited_texts)

        # E. Numbers in optimized text must appear in cited evidence.
        optimized_numbers = set(_NUMBER_RE.findall(action.optimized_text))
        if optimized_numbers:
            cited_numbers = {
                num for text in cited_texts for num in _NUMBER_RE.findall(text)
            }
            invented_numbers = optimized_numbers - cited_numbers
            if invented_numbers:
                issues.append(
                    ValidationIssue(
                        field="optimized_text",
                        message=(
                            "Numbers not found in cited evidence: "
                            f"{sorted(invented_numbers)}."
                        ),
                    )
                )

        # F. Skills in optimized text must be supported by cited evidence.
        if action.action_type in ("rewrite", "surface_existing_evidence"):
            optimized_skills = set(find_skills(action.optimized_text))
            if optimized_skills:
                cited_skills = set(find_skills(cited_text))
                unsupported_skills = optimized_skills - cited_skills
                if unsupported_skills:
                    issues.append(
                        ValidationIssue(
                            field="optimized_text",
                            message=(
                                "Skills not supported by cited evidence: "
                                f"{sorted(unsupported_skills)}."
                            ),
                        )
                    )

        # G. Experience section must preserve years present in cited evidence.
        if action.target_section == "experience":
            optimized_years = set(_YEAR_RE.findall(action.optimized_text))
            cited_years = {
                year for text in cited_texts for year in _YEAR_RE.findall(text)
            }
            new_years = optimized_years - cited_years
            if new_years:
                issues.append(
                    ValidationIssue(
                        field="optimized_text",
                        message=(
                            "Experience action introduces years not in cited "
                            f"evidence: {sorted(new_years)}."
                        ),
                    )
                )

        # H. Factual concepts in optimized text must be supported by cited evidence.
        if action.action_type in ("rewrite", "surface_existing_evidence"):
            optimized_concepts = extract_factual_concepts(action.optimized_text)
            original_concepts = (
                extract_factual_concepts(action.original_text)
                if action.original_text
                else frozenset()
            )
            new_concepts = optimized_concepts - original_concepts
            if new_concepts:
                cited_concepts: set[str] = set()
                for t in cited_texts:
                    cited_concepts |= extract_factual_concepts(t)
                unsupported = new_concepts - cited_concepts
                if unsupported:
                    issues.append(
                        ValidationIssue(
                            field="unsupported_concept",
                            message=(
                                "Optimized text introduces factual concepts "
                                "not found in cited evidence: "
                                f"{sorted(unsupported)}."
                            ),
                        )
                    )

        # I. Cross-role provenance guard for experience/projects sections.
        if (
            action.action_type in ("rewrite", "surface_existing_evidence")
            and action.target_section in ("experience", "projects")
        ):
            original_concepts = (
                extract_factual_concepts(action.original_text)
                if action.original_text
                else frozenset()
            )
            optimized_concepts = extract_factual_concepts(action.optimized_text)
            new_concepts = optimized_concepts - original_concepts
            if new_concepts:
                original_scopes: set[tuple] = set()
                for eid in action.evidence_ids:
                    ev = self._evidence_by_id.get(eid)
                    if ev is None:
                        continue
                    ev_concepts = extract_factual_concepts(ev.text)
                    if ev_concepts & original_concepts:
                        original_scopes.add(self._scope_key(ev))

                foreign_scopes: dict[tuple, list[str]] = {}
                for eid in action.evidence_ids:
                    ev = self._evidence_by_id.get(eid)
                    if ev is None:
                        continue
                    scope = self._scope_key(ev)
                    if original_scopes and scope in original_scopes:
                        continue
                    ev_concepts = extract_factual_concepts(ev.text)
                    contributed = sorted(ev_concepts & new_concepts)
                    if contributed:
                        foreign_scopes.setdefault(scope, []).extend(
                            contributed
                        )
                if foreign_scopes:
                    scope_labels = []
                    for scope in sorted(foreign_scopes):
                        section, entry_idx = scope
                        label = f"{section}[{entry_idx}]" if entry_idx else section
                        scope_labels.append(label)
                    issues.append(
                        ValidationIssue(
                            field="cross_scope_evidence",
                            message=(
                                "Optimized bullet merges factual concepts "
                                "from unrelated resume entries: "
                                f"{scope_labels}."
                            ),
                        )
                    )

        return issues

    def _scope_key(self, evidence: ResumeEvidence) -> tuple:
        return (evidence.section, evidence.metadata.get("entry_index"))
