import pytest
from pydantic import ValidationError

from app.models.schemas import (
    MatchRequest,
    MatchResult,
    RecommendationNarrative,
    ScoreComponent,
    SkillGap,
)


def test_match_result_score_bounds():
    assert MatchResult(overall_score=0, summary="s").overall_score == 0
    assert MatchResult(overall_score=100, summary="s").overall_score == 100
    with pytest.raises(ValidationError):
        MatchResult(overall_score=101, summary="s")
    with pytest.raises(ValidationError):
        MatchResult(overall_score=-1, summary="s")


def test_match_result_defaults():
    result = MatchResult(overall_score=42, summary="ok")
    assert result.matched_skills == []
    assert result.missing_skills == []
    assert result.strengths == []
    assert result.recommendations == []
    assert result.score_breakdown == []


def test_skill_gap_requires_both_fields():
    with pytest.raises(ValidationError):
        SkillGap(skill="Kubernetes")


def test_score_component_weight_bounds():
    ScoreComponent(name="required_skills", weight=1.0, score=100, detail="d")
    with pytest.raises(ValidationError):
        ScoreComponent(name="required_skills", weight=1.5, score=100, detail="d")
    with pytest.raises(ValidationError):
        ScoreComponent(name="required_skills", weight=0.5, score=101, detail="d")


def test_recommendation_narrative_defaults():
    narrative = RecommendationNarrative(summary="s")
    assert narrative.strengths == []
    assert narrative.recommendations == []


def test_match_request_accepts_normal_input():
    req = MatchRequest(resume_text="Hello world", job_description="Some job")
    assert req.resume_text == "Hello world"


def test_match_request_accepts_exactly_at_limit():
    text = "x" * 50000
    req = MatchRequest(resume_text=text, job_description=text)
    assert len(req.resume_text) == 50000
    assert len(req.job_description) == 50000


def test_match_request_rejects_over_limit():
    text = "x" * 50001
    with pytest.raises(ValidationError):
        MatchRequest(resume_text=text, job_description="ok")
    with pytest.raises(ValidationError):
        MatchRequest(resume_text="ok", job_description=text)


def test_match_request_rejects_empty():
    with pytest.raises(ValidationError):
        MatchRequest(resume_text="", job_description="ok")
    with pytest.raises(ValidationError):
        MatchRequest(resume_text="ok", job_description="")


def test_production_mode_disables_docs(monkeypatch):
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("APP_ENV", "production")
    try:
        from app.main import create_app

        app = create_app()
        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.openapi_url is None
    finally:
        get_settings.cache_clear()


def test_development_mode_enables_docs(monkeypatch):
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("APP_ENV", "development")
    try:
        from app.main import create_app

        app = create_app()
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"
    finally:
        get_settings.cache_clear()
