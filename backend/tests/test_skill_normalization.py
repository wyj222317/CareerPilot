"""Regression tests for deterministic skill normalization and compound matching."""

import pytest

from app.models.schemas import CandidateProfile, JobRequirements
from app.services.matching import MatchingService
from app.services.skill_normalization import (
    NormalizedRequirements,
    SkillGroup,
    SkillNormalizer,
)

NORMALIZER = SkillNormalizer()


def test_candidate_skill_aliases_are_normalized():
    profile = CandidateProfile(
        skills=[
            "Python",
            "JS",
            "Postgres",
            "OpenAI API",
            "DeepSeek API",
            "TS",
            "CI/CD",
        ]
    )
    normalized = NORMALIZER.normalize_candidate(profile)
    assert normalized.skills == [
        "ci/cd",
        "deepseek",
        "javascript",
        "openai",
        "postgresql",
        "python",
        "typescript",
    ]


def test_expert_level_python_matches_python():
    group = NORMALIZER._parse_requirement("Expert-level Python")
    assert group == SkillGroup(kind="all", skills=("python",), raw="Expert-level Python")


def test_or_group_extracts_both_options():
    group = NORMALIZER._parse_requirement("Production experience with LangGraph or LangChain")
    assert group is not None
    assert group.kind == "any"
    assert set(group.skills) == {"langchain", "langgraph"}


def test_and_group_extracts_both_required_options():
    group = NORMALIZER._parse_requirement("Strong API design skills with FastAPI and Pydantic")
    assert group is not None
    assert group.kind == "all"
    assert set(group.skills) == {"fastapi", "pydantic"}


def test_llm_api_phrase_extracts_three_providers_as_or():
    group = NORMALIZER._parse_requirement(
        "Hands-on experience integrating LLM APIs (OpenAI, DeepSeek, or Qwen)"
    )
    assert group is not None
    assert group.kind == "any"
    assert set(group.skills) == {"deepseek", "openai", "qwen"}


def test_experience_requirement_is_filtered_from_skills():
    group = NORMALIZER._parse_requirement(
        "4+ years of professional software engineering experience"
    )
    assert group is None


def test_education_requirement_is_filtered_from_skills():
    group = NORMALIZER._parse_requirement("Bachelor's degree in Computer Science")
    assert group is None


def test_unknown_phrase_is_preserved_as_unmapped_group():
    group = NORMALIZER._parse_requirement("Experience with EtherCAT")
    assert group is not None
    assert group.is_mapped is False
    assert group.skills == ()
    assert group.raw == "Experience with EtherCAT"
    assert "ethercat" in group.unmapped_terms


def test_compound_and_requires_both_skills():
    profile = CandidateProfile(skills=["FastAPI"])
    requirements = JobRequirements(
        required_skills=["Strong API design skills with FastAPI and Pydantic"]
    )
    result = MatchingService().calculate_match(profile, requirements)
    assert result.matched_skills == []
    assert any("pydantic" in gap.skill for gap in result.missing_skills)

    profile_with_both = CandidateProfile(skills=["FastAPI", "Pydantic"])
    result = MatchingService().calculate_match(profile_with_both, requirements)
    assert set(result.matched_skills) == {"fastapi", "pydantic"}
    assert result.missing_skills == []


def test_or_group_satisfied_by_either_skill():
    profile = CandidateProfile(skills=["LangGraph"])
    requirements = JobRequirements(
        required_skills=["Production experience with LangGraph or LangChain"]
    )
    result = MatchingService().calculate_match(profile, requirements)
    assert result.matched_skills == ["langgraph"]
    assert result.missing_skills == []


def test_or_group_unsatisfied_reports_readable_gap():
    profile = CandidateProfile(skills=["Python"])
    requirements = JobRequirements(
        required_skills=["Production experience with LangGraph or LangChain"]
    )
    result = MatchingService().calculate_match(profile, requirements)
    assert result.matched_skills == []
    assert any("LangGraph or LangChain" in gap.skill for gap in result.missing_skills)


def test_preferred_and_group_requires_both():
    profile = CandidateProfile(skills=["React", "TypeScript"])
    requirements = JobRequirements(
        preferred_skills=["Experience with React and TypeScript"]
    )
    result = MatchingService().calculate_match(profile, requirements)
    breakdown = {c.name: c for c in result.score_breakdown}
    assert breakdown["preferred_skills"].score == 100


def test_real_deepseek_failure_case_scores_high():
    """Regression for the observed real DeepSeek run.

    Before normalization the matcher compared raw requirement sentences against
    candidate skill names and reported 0 required-skill matches (overall 36).
    After normalization the same inputs match the atomic skills, but the
    hardening fix also surfaces unmapped atoms in partially-mapped AND groups
    (e.g. "relational data modeling"), so required skills no longer scores 100.
    """
    profile = CandidateProfile(
        name="Alex Chen",
        skills=[
            "Python",
            "LangChain",
            "LangGraph",
            "OpenAI API",
            "DeepSeek API",
            "FastAPI",
            "Pydantic",
            "PostgreSQL",
            "Docker",
            "AWS",
            "React",
            "TypeScript",
            "CI/CD",
        ],
        total_years_experience=5,
        experience=[],
        education=[],
    )
    requirements = JobRequirements(
        title="AI/Agent Engineer",
        required_skills=[
            "4+ years of professional software engineering experience",
            "Expert-level Python",
            "Production experience with LangGraph or LangChain",
            "Hands-on experience integrating LLM APIs (OpenAI, DeepSeek, or Qwen)",
            "Strong API design skills with FastAPI and Pydantic",
            "Comfortable with PostgreSQL and relational data modeling",
            "Experience deploying services with Docker and AWS",
        ],
        preferred_skills=[
            "Experience with React and TypeScript",
            "Familiarity with vector databases (Pinecone, Weaviate, or Chroma)",
            "Knowledge of CI/CD pipelines and infrastructure as code",
        ],
        min_years_experience=4,
        responsibilities=[],
        education_requirements=[],
    )

    result = MatchingService().calculate_match(profile, requirements)
    breakdown = {c.name: c for c in result.score_breakdown}

    # Required skill groups after filtering the experience sentence.
    assert len(NORMALIZER.normalize_requirements(requirements).required_groups) == 6
    # PostgreSQL group retains "relational data modeling" as an unsatisfiable atom.
    assert breakdown["required_skills"].score == 83
    assert set(result.matched_skills) == {
        "python",
        "langchain",
        "langgraph",
        "openai",
        "deepseek",
        "fastapi",
        "pydantic",
        "docker",
        "aws",
    }

    # Preferred: React+TypeScript AND satisfied; CI/CD AND retains
    # "infrastructure as code", so only the React/TypeScript group is satisfied.
    assert breakdown["preferred_skills"].score == 33
    profile_skills = set(NORMALIZER.normalize_candidate(profile).skills)
    preferred_matched = {
        skill
        for group in NORMALIZER.normalize_requirements(requirements).preferred_groups
        for skill in group.skills
        if skill in profile_skills
        and MatchingService._group_satisfied(group, profile_skills)
    }
    assert preferred_matched == {"react", "typescript"}

    # Experience requirement is captured by min_years_experience, not double-counted.
    assert breakdown["experience_fit"].score == 100
    assert breakdown["education_fit"].score == 100

    # Determinism.
    second = MatchingService().calculate_match(profile, requirements)
    assert result == second


def test_identical_inputs_produce_identical_scores_after_normalization():
    profile = CandidateProfile(skills=["Python", "Postgres"])
    requirements = JobRequirements(
        required_skills=["Expert-level Python", "Production experience with LangGraph or LangChain"],
        preferred_skills=["Experience with React and TypeScript"],
    )
    first = MatchingService().calculate_match(profile, requirements)
    second = MatchingService().calculate_match(profile, requirements)
    assert first == second
    assert first.overall_score == second.overall_score


def test_chinese_skill_aliases_normalize_to_canonical_skills():
    profile = CandidateProfile(
        skills=[
            "Python",
            "传感器",
            "雷达",
            "摄像头",
            "PLC",
            "工业机器人",
            "自动导引车",
        ]
    )
    normalized = NORMALIZER.normalize_candidate(profile)
    assert set(normalized.skills) == {
        "python",
        "sensor",
        "radar",
        "camera",
        "plc",
        "robotics",
        "agv",
    }


def test_chinese_experience_phrase_is_filtered_from_skills():
    group = NORMALIZER._parse_requirement("2年以上AI或机器人行业工作经验")
    assert group is None


def test_chinese_education_phrase_is_filtered_from_skills():
    group = NORMALIZER._parse_requirement("本科及以上学历，计算机科学或相关专业")
    assert group is None


def test_protocol_aliases_are_recognized():
    profile = CandidateProfile(skills=["ROS2", "CAN", "RS-485", "HTTP", "WebRTC"])
    normalized = NORMALIZER.normalize_candidate(profile)
    assert set(normalized.skills) == {"ros", "can", "rs485", "http", "webrtc"}
