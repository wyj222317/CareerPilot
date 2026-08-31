import operator
from typing import Annotated, TypedDict

from app.models.schemas import (
    CandidateProfile,
    EvidenceReport,
    JobRequirements,
    MatchResult,
    ResumeEvidence,
    ResumeOptimizationResult,
)
from app.services.evidence.vector_store import VectorStore


class AnalysisState(TypedDict, total=False):
    """Shared state flowing through the analysis graph.

    Nodes return partial updates; LangGraph merges them. `errors` uses an
    append-only reducer so any node can record a failure without clobbering
    errors from other nodes.
    """

    raw_resume_text: str
    raw_job_description: str
    candidate_profile: CandidateProfile | None
    job_requirements: JobRequirements | None
    match_result: MatchResult | None
    resume_evidence: list[ResumeEvidence] | None
    evidence_report: EvidenceReport | None
    vector_store: VectorStore | None
    resume_optimization: ResumeOptimizationResult | None
    errors: Annotated[list[str], operator.add]
