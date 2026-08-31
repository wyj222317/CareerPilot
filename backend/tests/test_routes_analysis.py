from app.services.llm import get_llm_service
from app.services.llm.base import LLMService


def test_match_returns_structured_result(client, sample_resume_pdf):
    resume_text = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.pdf", sample_resume_pdf, "application/pdf")},
    ).json()["extracted_text"]

    response = client.post(
        "/api/analysis/match",
        json={
            "resume_text": resume_text,
            "job_description": (
                "Senior Backend Engineer\n"
                "Required: Python, FastAPI, Kubernetes, GraphQL\n"
                "Preferred: AWS\n"
                "5+ years of experience."
            ),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["candidate"]["skills"]
    assert body["job_requirements"]["required_skills"]
    match = body["match"]
    assert 0 <= match["overall_score"] <= 100
    assert "python" in [skill.lower() for skill in match["matched_skills"]]
    assert "kubernetes" in [gap["skill"].lower() for gap in match["missing_skills"]]
    assert len(match["score_breakdown"]) == 5


def test_match_validates_request(client):
    response = client.post(
        "/api/analysis/match", json={"resume_text": "", "job_description": ""}
    )
    assert response.status_code == 422


def test_match_rejects_oversized_input(client):
    response = client.post(
        "/api/analysis/match",
        json={"resume_text": "x" * 50001, "job_description": "ok"},
    )
    assert response.status_code == 422


def test_match_returns_503_when_pipeline_fails(client):
    class FailingLLM(LLMService):
        def extract_candidate_profile(self, resume_text):
            raise RuntimeError("provider down")

        def extract_job_requirements(self, job_description):
            raise RuntimeError("provider down")

        def generate_recommendation(self, profile, requirements, match):
            raise RuntimeError("provider down")

    client.app.dependency_overrides[get_llm_service] = lambda: FailingLLM()
    try:
        response = client.post(
            "/api/analysis/match", json={"resume_text": "x", "job_description": "y"}
        )
        assert response.status_code == 503
        assert "parse_resume" in response.json()["detail"]
    finally:
        client.app.dependency_overrides.clear()
