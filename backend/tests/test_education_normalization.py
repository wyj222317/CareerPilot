"""Tests for deterministic education normalization, including Chinese degrees."""

from app.models.schemas import CandidateProfile, EducationItem, JobRequirements
from app.services.education_normalization import EducationNormalizer


def test_chinese_master_information_technology_maps_to_master_computing():
    profile = CandidateProfile(
        education=[
            EducationItem(degree="信息技术硕士", institution="示例大学", year="2024–2026")
        ]
    )
    norm = EducationNormalizer().normalize_candidate(profile)
    assert norm.level == "master"
    assert "computing" in norm.fields


def test_chinese_bachelor_mechanical_maps_to_bachelor_mechanical():
    profile = CandidateProfile(
        education=[
            EducationItem(
                degree="机械设计制造及其自动化｜工学学士",
                institution="示例理工大学",
                year="2018–2023",
            )
        ]
    )
    norm = EducationNormalizer().normalize_candidate(profile)
    assert norm.level == "bachelor"
    assert "mechanical" in norm.fields
    assert "robotics_automation" in norm.fields


def test_chinese_doctorate_maps_highest():
    profile = CandidateProfile(
        education=[EducationItem(degree="博士研究生", institution="示例大学")]
    )
    norm = EducationNormalizer().normalize_candidate(profile)
    assert norm.level == "doctorate"


def test_chinese_requirement_bachelor_computing_related():
    requirements = JobRequirements(
        education_requirements=[
            "本科及以上学历，计算机科学、自动化、机械工程或相关专业"
        ]
    )
    norms = EducationNormalizer().normalize_requirements(requirements)
    assert len(norms) == 1
    assert norms[0].min_level == "bachelor"
    assert "computing" in norms[0].acceptable_fields
    assert "robotics_automation" in norms[0].acceptable_fields
    assert "mechanical" in norms[0].acceptable_fields


def test_candidate_mechanical_satisfies_bachelor_mechanical_requirement():
    profile = CandidateProfile(
        education=[
            EducationItem(degree="工学学士，机械工程", institution="示例理工大学")
        ]
    )
    requirements = JobRequirements(
        education_requirements=["本科及以上学历，机械工程或相关专业"]
    )
    normalizer = EducationNormalizer()
    diag = normalizer.score(
        normalizer.normalize_candidate(profile),
        normalizer.normalize_requirements(requirements),
    )
    assert diag.score == 100


def test_candidate_level_none_scores_zero():
    profile = CandidateProfile(education=[])
    requirements = JobRequirements(
        education_requirements=["本科及以上学历，计算机科学或相关专业"]
    )
    normalizer = EducationNormalizer()
    diag = normalizer.score(
        normalizer.normalize_candidate(profile),
        normalizer.normalize_requirements(requirements),
    )
    assert diag.score == 0


def test_item_diagnostics_expose_raw_and_normalized():
    profile = CandidateProfile(
        education=[
            EducationItem(degree="信息技术硕士", institution="示例大学"),
            EducationItem(degree="工学学士，机械设计制造及其自动化"),
        ]
    )
    items = EducationNormalizer().candidate_item_diagnostics(profile)
    assert len(items) == 2
    assert items[0].raw_degree == "信息技术硕士"
    assert items[0].normalized_level == "master"
    assert "computing" in items[0].normalized_fields
    assert items[1].raw_degree == "工学学士，机械设计制造及其自动化"
    assert items[1].normalized_level == "bachelor"
    assert "mechanical" in items[1].normalized_fields
    assert "robotics_automation" in items[1].normalized_fields
