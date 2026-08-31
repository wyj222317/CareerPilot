from app.models.schemas import MatchAnalysisResponse
from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import EmbeddingService
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.graph import build_analysis_graph
from app.services.llm.base import LLMService
from app.services.matching import MatchingService


class AnalysisError(ValueError):
    """Raised when the analysis pipeline cannot produce a complete result."""


class MatchAnalysisService:
    """Runs the LangGraph analysis pipeline and maps state to a response."""

    def __init__(
        self,
        llm_service: LLMService,
        matching_service: MatchingService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store_factory=None,
        chunker: ResumeChunker | None = None,
        retriever: EvidenceRetriever | None = None,
    ) -> None:
        self._graph = build_analysis_graph(
            llm_service,
            matching_service,
            embedding_service,
            vector_store_factory,
            chunker,
            retriever,
        )

    def analyze(self, resume_text: str, job_description: str) -> MatchAnalysisResponse:
        state = self._graph.invoke(
            {
                "raw_resume_text": resume_text,
                "raw_job_description": job_description,
            }
        )
        errors = state.get("errors", [])
        profile = state.get("candidate_profile")
        requirements = state.get("job_requirements")
        match = state.get("match_result")
        if profile is None or requirements is None or match is None:
            raise AnalysisError("; ".join(errors) or "Analysis failed without a specific error.")
        return MatchAnalysisResponse(
            candidate=profile,
            job_requirements=requirements,
            match=match,
            optimization=state.get("resume_optimization"),
        )
