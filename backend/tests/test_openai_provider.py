import json
from types import SimpleNamespace

import pytest

from app.models.schemas import CandidateProfile, JobRequirements
from app.services.llm.base import LLMError
from app.services.llm.openai_compatible import OpenAICompatibleLLMService
from app.services.matching import MatchingService


def fake_client(responses):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, calls


def make_service(responses, **kwargs):
    client, calls = fake_client(responses)
    service = OpenAICompatibleLLMService(
        api_key="test-api-key",
        model="test-model",
        base_url="https://api.test/v1",
        client=client,
        **kwargs,
    )
    return service, calls


def test_valid_json_is_validated_into_model():
    payload = {"name": "Jane", "skills": ["Python"], "total_years_experience": 5}
    service, calls = make_service([json.dumps(payload)])

    profile = service.extract_candidate_profile("resume text")

    assert profile.name == "Jane"
    assert profile.skills == ["Python"]
    request = calls[0]
    assert request["model"] == "test-model"
    assert request["response_format"] == {"type": "json_object"}
    user_messages = [m for m in request["messages"] if m["role"] == "user"]
    assert "JSON Schema" in user_messages[0]["content"]


def test_invalid_json_retries_with_validation_feedback():
    service, calls = make_service(["not json at all", json.dumps({"title": "Engineer"})])

    requirements = service.extract_job_requirements("job text")

    assert requirements.title == "Engineer"
    assert len(calls) == 2
    retry_user_messages = [m for m in calls[1]["messages"] if m["role"] == "user"]
    assert any("failed validation" in m["content"] for m in retry_user_messages)


def test_schema_validation_failure_retries():
    service, calls = make_service(
        [json.dumps({"summary": 123}), json.dumps({"summary": "ok"})]
    )

    narrative = service.generate_recommendation(
        CandidateProfile(name="Jane"),
        JobRequirements(required_skills=["Python"]),
        MatchingService().calculate_match(
            CandidateProfile(name="Jane"), JobRequirements(required_skills=["Python"])
        ),
    )

    assert narrative.summary == "ok"
    assert len(calls) == 2


def test_exhausted_retries_raise_llm_error():
    service, _ = make_service(["nope", "still nope", "never"])

    with pytest.raises(LLMError, match="CandidateProfile"):
        service.extract_candidate_profile("resume text")


def test_generate_recommendation_serializes_context_into_prompt():
    service, calls = make_service(
        [json.dumps({"summary": "S", "strengths": ["A"], "recommendations": ["B"]})]
    )
    profile = CandidateProfile(name="Jane Doe", skills=["Python"])
    requirements = JobRequirements(required_skills=["Python"])
    match = MatchingService().calculate_match(profile, requirements)

    narrative = service.generate_recommendation(profile, requirements, match)

    assert narrative.summary == "S"
    user_message = [m for m in calls[0]["messages"] if m["role"] == "user"][0]["content"]
    assert "Jane Doe" in user_message
    assert "Python" in user_message
