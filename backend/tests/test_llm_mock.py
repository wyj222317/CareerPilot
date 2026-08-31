from app.services.llm.mock import MockLLMService
from app.services.matching import MatchingService

RESUME = """
Jane Doe
Senior Software Engineer
jane.doe@example.com

Skills: Python, JavaScript, React, FastAPI, SQL, Docker

Experience
Senior Software Engineer - TechCorp (2020-2023)
- Built REST APIs with FastAPI
Software Engineer - StartupXYZ (2018-2020)
- Developed React components

Education
B.S. Computer Science - State University (2018)
"""

JOB_DESCRIPTION = """
Senior Full-Stack Engineer

Required skills: Python, TypeScript, React, Kubernetes, GraphQL
Preferred: AWS, Docker
Responsibilities:
- Design and ship features
- Mentor junior engineers

5+ years of experience.
"""


def test_extracts_profile():
    profile = MockLLMService().extract_candidate_profile(RESUME)
    assert profile.name == "Jane Doe"
    assert profile.email == "jane.doe@example.com"
    skills = {skill.lower() for skill in profile.skills}
    assert {"python", "react", "fastapi", "docker"} <= skills
    assert [item.company for item in profile.experience] == ["TechCorp", "StartupXYZ"]
    assert profile.experience[0].highlights == ["Built REST APIs with FastAPI"]
    assert profile.total_years_experience == 5
    assert profile.education[0].degree == "B.S. Computer Science"
    assert profile.education[0].institution == "State University"


def test_extracts_requirements():
    requirements = MockLLMService().extract_job_requirements(JOB_DESCRIPTION)
    required = {skill.lower() for skill in requirements.required_skills}
    preferred = {skill.lower() for skill in requirements.preferred_skills}
    assert {"python", "kubernetes", "graphql"} <= required
    assert {"aws", "docker"} <= preferred
    assert requirements.min_years_experience == 5
    assert requirements.responsibilities == [
        "Design and ship features",
        "Mentor junior engineers",
    ]
    assert requirements.education_requirements == []


def test_generate_recommendation_returns_narrative():
    service = MockLLMService()
    profile = service.extract_candidate_profile(RESUME)
    requirements = service.extract_job_requirements(JOB_DESCRIPTION)
    match = MatchingService().calculate_match(profile, requirements)

    narrative = service.generate_recommendation(profile, requirements, match)

    assert narrative.summary
    assert narrative.strengths
    assert narrative.recommendations
