from typing import Literal

from pydantic import BaseModel, Field

SupportLevel = Literal["supported", "partially_supported", "unsupported"]


class ExperienceItem(BaseModel):
    title: str
    company: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    highlights: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    degree: str
    institution: str | None = None
    year: str | None = None


class CandidateProfile(BaseModel):
    """Structured representation of a candidate, parsed from their resume."""

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    total_years_experience: int | None = Field(default=None, ge=0)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)


class JobRequirements(BaseModel):
    """Structured representation of what a job posting asks for."""

    title: str | None = None
    company: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    behavioral_competencies: list[str] = Field(default_factory=list)
    compliance_requirements: list[str] = Field(default_factory=list)
    min_years_experience: int | None = Field(default=None, ge=0)
    responsibilities: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)


class SkillGap(BaseModel):
    skill: str
    suggestion: str


class ScoreComponent(BaseModel):
    """One rubric dimension contributing to the overall match score."""

    name: str
    weight: float = Field(ge=0.0, le=1.0)
    score: int = Field(ge=0, le=100)
    detail: str


class ResumeEvidence(BaseModel):
    """A verbatim chunk of resume text traceable to a specific section."""

    id: str
    text: str
    section: str
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)


class EvidenceMatch(BaseModel):
    """One piece of resume evidence retrieved for a claim."""

    evidence_id: str
    text: str
    section: str
    score: float = Field(ge=0.0, le=1.0)


class ClaimEvidence(BaseModel):
    """A claim (skill, strength, responsibility) with its retrieved evidence."""

    claim: str
    claim_type: Literal["required_skill", "preferred_skill", "strength", "responsibility"]
    support_level: SupportLevel
    evidence: list[EvidenceMatch] = Field(default_factory=list)


class EvidenceReport(BaseModel):
    """Aggregated evidence retrieval results for all claims."""

    claims: list[ClaimEvidence] = Field(default_factory=list)
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0


class MatchResult(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    summary: str
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[SkillGap] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    score_breakdown: list[ScoreComponent] = Field(default_factory=list)
    behavioral_competencies: list[str] = Field(default_factory=list)
    compliance_requirements: list[str] = Field(default_factory=list)
    evidence: EvidenceReport | None = None


class RecommendationNarrative(BaseModel):
    """Natural-language output of the recommendation step.

    The LLM fills only narrative fields; scores and skill comparisons are
    computed deterministically by MatchingService and never modified here.
    """

    summary: str
    strengths: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


OptimizationActionType = Literal[
    "rewrite", "reorder", "surface_existing_evidence", "gap_warning"
]


class OptimizationAction(BaseModel):
    """One evidence-constrained resume optimization suggestion.

    Action type semantics:
    - rewrite: rewrite an existing resume bullet. Requires original_text,
      optimized_text, and at least one evidence_id.
    - surface_existing_evidence: present existing evidence more prominently.
      Requires optimized_text and at least one evidence_id. original_text is
      required when tied to an existing bullet; it may be omitted only for a
      new summary/bullet built entirely from cited evidence.
    - reorder: change the order of existing content. Must identify the existing
      content (original_text/evidence_ids) and must not invent new factual text.
    - gap_warning: belongs in the gap_warnings list, not in actions.

    Every rewrite/surface/reorder action must cite one or more ResumeEvidence IDs
    and stay within the facts present in that evidence.
    """

    action_type: OptimizationActionType = Field(
        description=(
            "Allowed values: rewrite, surface_existing_evidence, reorder. "
            "gap_warning belongs in gap_warnings, not actions."
        )
    )
    target_section: str = Field(
        description="Resume section being optimized, e.g. summary, skills, experience, education."
    )
    original_text: str | None = Field(
        default=None,
        description=(
            "Existing resume text being changed. Required for rewrite and for "
            "surface_existing_evidence when tied to an existing bullet. May be "
            "omitted for surface_existing_evidence only when the action is a new "
            "surfaced summary/bullet built entirely from cited evidence."
        ),
    )
    optimized_text: str | None = Field(
        default=None,
        description=(
            "Suggested resume-facing text. Required for rewrite and "
            "surface_existing_evidence. Must be grounded in the cited evidence_ids."
        ),
    )
    rationale: str = Field(
        description="Why this action is suggested and how it respects the evidence boundary."
    )
    target_requirements: list[str] = Field(
        default_factory=list,
        description="Job requirements this action addresses.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "ResumeEvidence IDs that justify the optimized_text. Required for "
            "rewrite, surface_existing_evidence, and reorder."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0,
        description="Confidence in the suggestion; must be between 0 and 1.",
    )


class GapWarning(BaseModel):
    """A requirement that cannot be addressed without inventing facts."""

    requirement: str
    rationale: str


class RewrittenBullet(BaseModel):
    """A rewritten resume bullet constrained by cited evidence."""

    section: str
    original: str | None = None
    rewritten: str
    evidence_ids: list[str] = Field(default_factory=list)
    target_requirements: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    """A post-generation validation failure for an optimization action."""

    field: str
    message: str


class ResumeOptimizationResult(BaseModel):
    """Structured, auditable resume optimization output."""

    actions: list[OptimizationAction] = Field(default_factory=list)
    rewritten_bullets: list[RewrittenBullet] = Field(default_factory=list)
    gap_warnings: list[GapWarning] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    summary: str


class ResumeUploadResponse(BaseModel):
    filename: str
    page_count: int
    character_count: int
    extracted_text: str


class MatchRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=50000)
    job_description: str = Field(min_length=1, max_length=50000)


class MatchAnalysisResponse(BaseModel):
    candidate: CandidateProfile
    job_requirements: JobRequirements
    match: MatchResult
    optimization: ResumeOptimizationResult | None = None
