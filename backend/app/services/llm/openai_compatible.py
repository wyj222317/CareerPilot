import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.models.schemas import (
    CandidateProfile,
    EvidenceReport,
    GapWarning,
    JobRequirements,
    MatchResult,
    OptimizationAction,
    RecommendationNarrative,
    ResumeEvidence,
    ResumeOptimizationResult,
)
from app.prompts import evidence, job_fit, optimization
from app.services.llm.base import LLMError, LLMService

ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_RETRIES = 2


class OpenAICompatibleLLMService(LLMService):
    """LLM service for any OpenAI-compatible chat-completions API.

    Works with OpenAI, DeepSeek, and Qwen (DashScope compatible mode) by
    pointing the OpenAI SDK at the provider's base URL. Structured output is
    requested via `response_format={"type": "json_object"}` plus the target
    Pydantic model's JSON schema embedded in the user message; responses are
    validated with Pydantic and retried with the validation error fed back
    to the model.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_retries: int = DEFAULT_MAX_OUTPUT_RETRIES,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._client = client
        self.model = model
        self.base_url = base_url
        self._temperature = temperature
        self._max_output_retries = max_output_retries

    def extract_candidate_profile(self, resume_text: str) -> CandidateProfile:
        return self._structured_complete(
            job_fit.EXTRACT_CANDIDATE_PROFILE_SYSTEM,
            job_fit.EXTRACT_CANDIDATE_PROFILE_USER.format(resume_text=resume_text),
            CandidateProfile,
        )

    def extract_job_requirements(self, job_description: str) -> JobRequirements:
        return self._structured_complete(
            job_fit.EXTRACT_JOB_REQUIREMENTS_SYSTEM,
            job_fit.EXTRACT_JOB_REQUIREMENTS_USER.format(job_description=job_description),
            JobRequirements,
        )

    def generate_recommendation(
        self, profile: CandidateProfile, requirements: JobRequirements, match: MatchResult
    ) -> RecommendationNarrative:
        user = job_fit.GENERATE_RECOMMENDATION_USER.format(
            profile_json=profile.model_dump_json(indent=2),
            requirements_json=requirements.model_dump_json(indent=2),
            match_json=match.model_dump_json(indent=2),
        )
        return self._structured_complete(
            job_fit.GENERATE_RECOMMENDATION_SYSTEM, user, RecommendationNarrative
        )

    def generate_grounded_recommendation(
        self,
        profile: CandidateProfile,
        requirements: JobRequirements,
        match: MatchResult,
        evidence_report: EvidenceReport | None = None,
    ) -> RecommendationNarrative:
        evidence_json = (
            evidence_report.model_dump_json(indent=2) if evidence_report else "null"
        )
        user = evidence.GENERATE_GROUNDED_RECOMMENDATION_USER.format(
            profile_json=profile.model_dump_json(indent=2),
            requirements_json=requirements.model_dump_json(indent=2),
            match_json=match.model_dump_json(indent=2),
            evidence_json=evidence_json,
        )
        return self._structured_complete(
            evidence.GENERATE_GROUNDED_RECOMMENDATION_SYSTEM,
            user,
            RecommendationNarrative,
        )

    def generate_resume_optimization(
        self,
        requirements: JobRequirements,
        match: MatchResult,
        evidence_report: EvidenceReport,
        policy_actions: list[OptimizationAction],
        gap_warnings: list[GapWarning],
        relevant_evidence: list[ResumeEvidence],
    ) -> ResumeOptimizationResult:
        user = optimization.GENERATE_RESUME_OPTIMIZATION_USER.format(
            requirements_json=requirements.model_dump_json(indent=2),
            match_json=match.model_dump_json(indent=2),
            evidence_json=evidence_report.model_dump_json(indent=2),
            policy_actions_json=json.dumps(
                [a.model_dump() for a in policy_actions], indent=2
            ),
            gap_warnings_json=json.dumps(
                [g.model_dump() for g in gap_warnings], indent=2
            ),
            relevant_evidence_json=json.dumps(
                [e.model_dump() for e in relevant_evidence], indent=2
            ),
        )
        return self._structured_complete(
            optimization.GENERATE_RESUME_OPTIMIZATION_SYSTEM,
            user,
            ResumeOptimizationResult,
        )

    def _structured_complete(
        self, system: str, user: str, model_cls: type[ModelT]
    ) -> ModelT:
        schema = json.dumps(model_cls.model_json_schema())
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"{user}\n\nRespond with a single JSON object that validates "
                    f"against this JSON Schema:\n{schema}"
                ),
            },
        ]
        last_error: Exception | None = None
        attempts = self._max_output_retries + 1
        for _ in range(attempts):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            try:
                return model_cls.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"That response failed validation: {exc}\n"
                            "Return only a corrected JSON object matching the schema."
                        ),
                    },
                ]
        raise LLMError(
            f"Provider returned no valid {model_cls.__name__} JSON after "
            f"{attempts} attempt(s): {last_error}"
        )
