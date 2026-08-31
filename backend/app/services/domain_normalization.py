"""Deterministic, explainable domain-family normalization.

Maps free-text signals (skills, responsibilities, summaries, experience) onto
a small set of canonical domain families and computes exact + adjacent overlap.
No embeddings, no LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import CandidateProfile, JobRequirements

DOMAIN_FAMILIES = {
    "robotics",
    "industrial_automation",
    "data_acquisition",
    "software_ai",
}

# signal -> families it supports. Longest signals first for regex alternation.
DOMAIN_SIGNALS: dict[str, tuple[str, ...]] = {
    # Industrial automation
    "industrial automation": ("industrial_automation",),
    "factory automation": ("industrial_automation",),
    "manufacturing automation": ("industrial_automation",),
    "工业自动化": ("industrial_automation",),
    "自动化": ("industrial_automation",),
    "plc": ("industrial_automation",),
    "可编程逻辑控制器": ("industrial_automation",),
    "mes": ("industrial_automation",),
    "制造执行系统": ("industrial_automation",),
    "scada": ("industrial_automation",),
    "agv": ("industrial_automation", "robotics"),
    "自动导引车": ("industrial_automation", "robotics"),
    "stk": ("industrial_automation",),
    "oht": ("industrial_automation",),
    # Robotics
    "robotics": ("robotics",),
    "robot": ("robotics",),
    "robots": ("robotics",),
    "机器人": ("robotics",),
    "工业机器人": ("robotics", "industrial_automation"),
    "机械臂": ("robotics",),
    "协作机器人": ("robotics",),
    "ros": ("robotics",),
    "ros2": ("robotics",),
    "机器人操作系统": ("robotics",),
    "motion control": ("robotics",),
    "autonomous systems": ("robotics",),
    # Data acquisition
    "data acquisition": ("data_acquisition",),
    "data collection": ("data_acquisition",),
    "数据采集": ("data_acquisition",),
    "motion capture": ("data_acquisition",),
    "运动捕捉": ("data_acquisition",),
    "telemetry": ("data_acquisition",),
    "sensor": ("data_acquisition", "robotics"),
    "sensors": ("data_acquisition", "robotics"),
    "传感器": ("data_acquisition", "robotics"),
    "camera": ("data_acquisition",),
    "摄像头": ("data_acquisition",),
    "相机": ("data_acquisition",),
    "radar": ("data_acquisition",),
    "雷达": ("data_acquisition",),
    "lidar": ("data_acquisition",),
    "激光雷达": ("data_acquisition",),
    "imu": ("data_acquisition", "robotics"),
    "惯性测量单元": ("data_acquisition", "robotics"),
    # Software / AI
    "computer vision": ("software_ai", "data_acquisition"),
    "计算机视觉": ("software_ai", "data_acquisition"),
    "machine learning": ("software_ai",),
    "机器学习": ("software_ai",),
    "deep learning": ("software_ai",),
    "深度学习": ("software_ai",),
    "data science": ("software_ai",),
    "数据科学": ("software_ai",),
    "artificial intelligence": ("software_ai",),
    "人工智能": ("software_ai",),
    "ai": ("software_ai",),
    "llm": ("software_ai",),
    "rag": ("software_ai",),
    "python": ("software_ai",),
    "c": ("software_ai",),
    "linux": ("software_ai",),
}

_DOMAIN_RE = re.compile(
    r"(?<![a-z0-9])"
    + "(" + "|".join(re.escape(signal) for signal in sorted(DOMAIN_SIGNALS, key=len, reverse=True)) + ")"
    + r"(?![a-z0-9])",
    re.IGNORECASE,
)

# Adjacent families count as partial evidence. Direction matters: a candidate
# family can satisfy a job family if they are adjacent.
DOMAIN_ADJACENCY: dict[str, set[str]] = {
    "industrial_automation": {"robotics"},
    "robotics": {"industrial_automation", "data_acquisition"},
    "data_acquisition": {"robotics"},
    "software_ai": set(),
}


@dataclass(frozen=True)
class DomainDiagnostics:
    candidate_families: frozenset[str]
    job_families: frozenset[str]
    exact_matches: frozenset[str]
    adjacent_matches: frozenset[tuple[str, str]]  # (candidate_family, job_family)
    score: int


class DomainNormalizer:
    """Map candidate and job text onto domain families and score overlap."""

    def normalize_candidate(self, profile: CandidateProfile) -> frozenset[str]:
        """Collect domain families evidenced by the candidate."""
        parts: list[str] = []
        parts.extend(profile.skills)
        if profile.summary:
            parts.append(profile.summary)
        for exp in profile.experience:
            if exp.title:
                parts.append(exp.title)
            parts.extend(exp.highlights)
        return self._families(" ".join(parts))

    def normalize_job(self, requirements: JobRequirements) -> frozenset[str]:
        """Collect domain families required by the job responsibilities."""
        return self._families(" ".join(requirements.responsibilities))

    @staticmethod
    def _families(text: str) -> frozenset[str]:
        families: set[str] = set()
        for match in _DOMAIN_RE.finditer(text.lower()):
            families.update(DOMAIN_SIGNALS[match.group(1).lower()])
        return frozenset(families)

    def score(
        self,
        candidate_families: frozenset[str],
        job_families: frozenset[str],
    ) -> DomainDiagnostics:
        exact = candidate_families & job_families
        adjacent: set[tuple[str, str]] = set()
        for job_family in job_families - exact:
            for cand_family in candidate_families:
                if cand_family in DOMAIN_ADJACENCY.get(job_family, set()):
                    adjacent.add((cand_family, job_family))
                    break

        if not job_families:
            score = 100
        else:
            score = min(
                100,
                round(100 * (len(exact) + 0.5 * len(adjacent)) / len(job_families)),
            )

        return DomainDiagnostics(
            candidate_families=candidate_families,
            job_families=job_families,
            exact_matches=exact,
            adjacent_matches=frozenset(adjacent),
            score=score,
        )
