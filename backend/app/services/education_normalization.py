"""Deterministic education normalization for rubric scoring.

Parses degree level and field family separately so that "M.S. Information
Technology" can satisfy "Bachelor's degree or above in Computer Science ... or
related field" without requiring literal token overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import CandidateProfile, JobRequirements
from app.services.text_normalization import contains_cjk

DEGREE_ORDER = {
    "certificate": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "doctorate": 5,
}

DEGREE_PATTERNS: dict[str, list[str]] = {
    "certificate": ["certificate", "certification", "专科", "大专"],
    "associate": ["associate"],
    "bachelor": [
        "bachelor",
        "b\\.s\\.?",
        "b\\.a\\.?",
        "bs",
        "ba",
        "undergraduate",
        # Chinese
        "学士",
        "本科",
        "工学学士",
        "理学学士",
        "文学学士",
        "大学本科",
    ],
    "master": [
        "master",
        "m\\.s\\.?",
        "m\\.a\\.?",
        "ms",
        "ma",
        "mba",
        "graduate",
        # Chinese
        "硕士",
        "研究生",
        "工学硕士",
        "工程硕士",
        "理学硕士",
        "硕士研究生",
    ],
    "doctorate": [
        "doctorate",
        "doctoral",
        "ph\\.d\\.?",
        "phd",
        # Chinese
        "博士",
        "博士研究生",
    ],
}

FIELD_PATTERNS: dict[str, list[str]] = {
    "computing": [
        "computer science",
        "computer engineering",
        "information technology",
        "software engineering",
        "information systems",
        "computing",
        # Chinese
        "信息技术",
        "计算机科学",
        "计算机",
        "软件工程",
        "信息系统",
        "计算机科学与技术",
    ],
    "ai_data": [
        "artificial intelligence",
        "ai",
        "data science",
        "data engineering",
        "machine learning",
        "deep learning",
        # Chinese
        "人工智能",
        "数据科学",
        "机器学习",
        "深度学习",
    ],
    "robotics_automation": [
        "robotics",
        "automation",
        "mechatronics",
        # Chinese
        "自动化",
        "机器人工程",
        "机电一体化",
        "机械设计制造及其自动化",
        "控制工程",
    ],
    "electrical_electronics": [
        "electrical engineering",
        "electronics",
        # Chinese
        "电气工程",
        "电子工程",
        "电子信息",
    ],
    "mechanical": [
        "mechanical engineering",
        "mechanical",
        # Chinese
        "机械工程",
        "机械",
        "机械设计制造及其自动化",
    ],
    "business": [
        "business",
        "mba",
    ],
}


@dataclass(frozen=True)
class EducationRequirementNorm:
    raw: str
    min_level: str | None
    acceptable_fields: frozenset[str]


@dataclass(frozen=True)
class EducationCandidateNorm:
    level: str | None
    fields: frozenset[str]


@dataclass(frozen=True)
class EducationItemDiagnostics:
    raw_degree: str
    normalized_level: str | None
    raw_field_sources: tuple[str, ...]
    normalized_fields: frozenset[str]


@dataclass(frozen=True)
class EducationDiagnostics:
    candidate_level: str | None
    candidate_fields: frozenset[str]
    requirements: tuple[EducationRequirementNorm, ...]
    level_match: bool
    field_match: bool
    score: int
    item_diagnostics: tuple[EducationItemDiagnostics, ...] = ()


def _education_pattern_regex(pattern: str) -> str:
    """Build a matcher for one education pattern.

    ASCII patterns may contain intentional regex escapes (e.g. ``m\.s\.?``),
    so they are wrapped in word boundaries without re-escaping. CJK patterns
    are matched as literal substrings.
    """
    if contains_cjk(pattern):
        return re.escape(pattern)
    return rf"\b(?:{pattern})\b"


def _detect_degree_level(text: str) -> str | None:
    lowered = text.lower()
    for level in ("doctorate", "master", "bachelor", "associate", "certificate"):
        for pattern in DEGREE_PATTERNS[level]:
            if re.search(_education_pattern_regex(pattern), lowered):
                return level
    return None


def _detect_field_families(text: str) -> frozenset[str]:
    lowered = text.lower()
    found: set[str] = set()
    for family, patterns in FIELD_PATTERNS.items():
        if any(
            re.search(_education_pattern_regex(pattern), lowered) for pattern in patterns
        ):
            found.add(family)
    return frozenset(found)


class EducationNormalizer:
    """Normalize candidate education and JD requirements for deterministic scoring."""

    def normalize_candidate(self, profile: CandidateProfile) -> EducationCandidateNorm:
        if not profile.education:
            return EducationCandidateNorm(level=None, fields=frozenset())
        best_level: str | None = None
        best_order = 0
        all_fields: set[str] = set()
        for item in profile.education:
            level = _detect_degree_level(item.degree)
            if level and DEGREE_ORDER[level] > best_order:
                best_level = level
                best_order = DEGREE_ORDER[level]
            all_fields.update(_detect_field_families(item.degree))
        return EducationCandidateNorm(level=best_level, fields=frozenset(all_fields))

    def candidate_item_diagnostics(
        self, profile: CandidateProfile
    ) -> tuple[EducationItemDiagnostics, ...]:
        """Return per-education-entry normalization details for diagnostics."""
        items: list[EducationItemDiagnostics] = []
        for item in profile.education:
            level = _detect_degree_level(item.degree)
            fields = _detect_field_families(item.degree)
            items.append(
                EducationItemDiagnostics(
                    raw_degree=item.degree,
                    normalized_level=level,
                    raw_field_sources=(item.degree,),
                    normalized_fields=fields,
                )
            )
        return tuple(items)

    def normalize_requirements(
        self, requirements: JobRequirements
    ) -> tuple[EducationRequirementNorm, ...]:
        norms: list[EducationRequirementNorm] = []
        for req in requirements.education_requirements:
            min_level = _detect_degree_level(req)
            fields = _detect_field_families(req)
            # If the requirement says "or above" without a specific upper bound,
            # the min_level is the lowest acceptable level.
            norms.append(
                EducationRequirementNorm(
                    raw=req,
                    min_level=min_level,
                    acceptable_fields=fields,
                )
            )
        return tuple(norms)

    def score(
        self,
        candidate: EducationCandidateNorm,
        requirements: tuple[EducationRequirementNorm, ...],
        item_diagnostics: tuple[EducationItemDiagnostics, ...] | None = None,
    ) -> EducationDiagnostics:
        if not requirements:
            return EducationDiagnostics(
                candidate_level=candidate.level,
                candidate_fields=candidate.fields,
                requirements=requirements,
                level_match=True,
                field_match=True,
                score=100,
                item_diagnostics=item_diagnostics or (),
            )

        if candidate.level is None:
            return EducationDiagnostics(
                candidate_level=None,
                candidate_fields=candidate.fields,
                requirements=requirements,
                level_match=False,
                field_match=False,
                score=0,
                item_diagnostics=item_diagnostics or (),
            )

        level_scores: list[bool] = []
        field_scores: list[bool] = []
        for req in requirements:
            if req.min_level is None or DEGREE_ORDER[candidate.level] >= DEGREE_ORDER[req.min_level]:
                level_scores.append(True)
            else:
                level_scores.append(False)

            if not req.acceptable_fields or (candidate.fields & req.acceptable_fields):
                field_scores.append(True)
            else:
                field_scores.append(False)

        level_match = all(level_scores)
        field_match = all(field_scores)
        score = round(100 * (level_match + field_match) / 2)

        return EducationDiagnostics(
            candidate_level=candidate.level,
            candidate_fields=candidate.fields,
            requirements=requirements,
            level_match=level_match,
            field_match=field_match,
            score=score,
            item_diagnostics=item_diagnostics or (),
        )
