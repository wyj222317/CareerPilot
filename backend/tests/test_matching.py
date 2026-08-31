from app.models.schemas import (
    CandidateProfile,
    EducationItem,
    ExperienceItem,
    JobRequirements,
)
from app.services.matching import MatchingService

PROFILE = CandidateProfile(
    name="Jane Doe",
    skills=["Python", "React", "FastAPI", "docker"],
    total_years_experience=5,
    experience=[
        ExperienceItem(
            title="Senior Software Engineer",
            company="TechCorp",
            highlights=["Built and operated REST APIs", "Mentored junior engineers"],
        )
    ],
    education=[EducationItem(degree="B.S. Computer Science", institution="State University")],
)

REQUIREMENTS = JobRequirements(
    title="Senior Full-Stack Engineer",
    required_skills=["Python", "TypeScript", "React", "Kubernetes"],
    preferred_skills=["Docker", "AWS"],
    min_years_experience=3,
    responsibilities=["Build and operate REST APIs", "Mentor junior engineers"],
    education_requirements=["Bachelor's degree in Computer Science"],
)

EXPECTED_WEIGHTS = [
    ("required_skills", 0.40),
    ("domain_fit", 0.20),
    ("experience_fit", 0.20),
    ("education_fit", 0.10),
    ("preferred_skills", 0.10),
]


def by_name(result):
    return {component.name: component for component in result.score_breakdown}


def test_rubric_components_and_weights():
    result = MatchingService().calculate_match(PROFILE, REQUIREMENTS)
    assert [(c.name, c.weight) for c in result.score_breakdown] == EXPECTED_WEIGHTS


def test_overall_score_is_weighted_sum():
    result = MatchingService().calculate_match(PROFILE, REQUIREMENTS)
    expected = round(sum(c.weight * c.score for c in result.score_breakdown))
    assert result.overall_score == expected
    assert 0 <= result.overall_score <= 100


def test_skills_partition_is_case_insensitive():
    result = MatchingService().calculate_match(PROFILE, REQUIREMENTS)
    matched = {skill.lower() for skill in result.matched_skills}
    missing = {gap.skill.lower() for gap in result.missing_skills}
    assert matched == {"python", "react"}
    assert missing == {"typescript", "kubernetes"}
    assert not matched & missing
    assert all(gap.suggestion for gap in result.missing_skills)


def test_identical_inputs_produce_identical_results():
    service = MatchingService()
    first = service.calculate_match(PROFILE, REQUIREMENTS)
    second = service.calculate_match(PROFILE, REQUIREMENTS)
    assert first == second


def test_empty_requirements_score_components_full():
    result = MatchingService().calculate_match(PROFILE, JobRequirements())
    components = by_name(result)
    assert components["required_skills"].score == 100
    assert components["preferred_skills"].score == 100
    assert components["experience_fit"].score == 100
    assert components["education_fit"].score == 100
    assert components["domain_fit"].score == 100


def test_unknown_experience_years_scores_neutral():
    profile = PROFILE.model_copy(update={"total_years_experience": None})
    result = MatchingService().calculate_match(profile, REQUIREMENTS)
    assert by_name(result)["experience_fit"].score == 50


def test_exceeding_required_years_caps_at_100():
    profile = PROFILE.model_copy(update={"total_years_experience": 30})
    result = MatchingService().calculate_match(profile, REQUIREMENTS)
    assert by_name(result)["experience_fit"].score == 100


def test_missing_candidate_education_scores_zero():
    profile = PROFILE.model_copy(update={"education": []})
    result = MatchingService().calculate_match(profile, REQUIREMENTS)
    assert by_name(result)["education_fit"].score == 0


def test_components_carry_explanations():
    result = MatchingService().calculate_match(PROFILE, REQUIREMENTS)
    assert all(component.detail for component in result.score_breakdown)
