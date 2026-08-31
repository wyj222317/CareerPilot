"""LangGraph node factories and routing functions.

Nodes are thin state adapters: they call an LLMService method or the
deterministic MatchingService, write their result into the shared state,
and record failures in `errors` instead of raising. Routing functions are
pure predicates over the state.
"""

from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import EmbeddingService
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import VectorStore
from app.services.llm.base import LLMService
from app.services.matching import MatchingService
from app.services.optimization.optimizer import ResumeOptimizer
from app.services.optimization.validator import OptimizationValidator


def make_parse_resume_node(llm_service: LLMService):
    def parse_resume(state) -> dict:
        try:
            profile = llm_service.extract_candidate_profile(state["raw_resume_text"])
        except Exception as exc:
            return {"errors": [f"parse_resume: {exc}"]}
        warnings: list[str] = []
        raw_text = state.get("raw_resume_text") or ""
        if (
            len(raw_text) > 100
            and not profile.skills
            and not profile.experience
            and not profile.education
        ):
            warnings.append(
                "parse_resume: resume extraction returned no skills, experience, or "
                "education from substantial input. With the heuristic mock provider, "
                "this usually means the resume format is unsupported; connect a real "
                "LLM provider for production-quality extraction."
            )
        result: dict = {"candidate_profile": profile}
        if warnings:
            result["errors"] = warnings
        return result

    return parse_resume


def make_analyze_job_description_node(llm_service: LLMService):
    def analyze_job_description(state) -> dict:
        try:
            requirements = llm_service.extract_job_requirements(state["raw_job_description"])
        except Exception as exc:
            return {"errors": [f"analyze_job_description: {exc}"]}
        warnings: list[str] = []
        raw_text = state.get("raw_job_description") or ""
        if (
            len(raw_text) > 100
            and not requirements.required_skills
            and not requirements.responsibilities
        ):
            warnings.append(
                "analyze_job_description: no required skills or responsibilities "
                "extracted from substantial job description. With the heuristic mock "
                "provider, this usually means the JD format is unsupported; connect a "
                "real LLM provider for production-quality extraction."
            )
        result: dict = {"job_requirements": requirements}
        if warnings:
            result["errors"] = warnings
        return result

    return analyze_job_description


def make_calculate_match_node(matching_service: MatchingService):
    def calculate_match(state) -> dict:
        try:
            match = matching_service.calculate_match(
                state["candidate_profile"], state["job_requirements"]
            )
        except Exception as exc:
            return {"errors": [f"calculate_match: {exc}"]}
        return {"match_result": match}

    return calculate_match


def make_build_evidence_index_node(
    chunker: ResumeChunker,
    embedder: EmbeddingService,
    vector_store_factory,
):
    """Build the evidence index from raw resume text, embed, populate vector store."""
    def build_evidence_index(state) -> dict:
        try:
            raw_text = state["raw_resume_text"]
            profile = state.get("candidate_profile")
            chunks = chunker.chunk(raw_text, profile)
            if not chunks:
                return {"resume_evidence": []}

            texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed(texts)
            store = vector_store_factory()
            store.add(chunks, vectors)

            return {"resume_evidence": chunks, "vector_store": store}
        except Exception as exc:
            return {"errors": [f"build_evidence_index: {exc}"]}

    return build_evidence_index


def make_retrieve_evidence_node(
    retriever: EvidenceRetriever,
):
    """Retrieve evidence for claims from the vector store."""
    def retrieve_evidence(state) -> dict:
        try:
            store = state.get("vector_store")
            if store is None:
                return {"evidence_report": None}

            requirements = state["job_requirements"]
            match = state["match_result"]
            report = retriever.retrieve(requirements, match, store)
            return {"evidence_report": report}
        except Exception as exc:
            return {"errors": [f"retrieve_evidence: {exc}"]}

    return retrieve_evidence


def make_generate_grounded_recommendation_node(llm_service: LLMService):
    """Generate grounded recommendation using evidence."""
    def generate_grounded_recommendation(state) -> dict:
        try:
            profile = state["candidate_profile"]
            requirements = state["job_requirements"]
            match = state["match_result"]
            evidence_report = state.get("evidence_report")

            narrative = llm_service.generate_grounded_recommendation(
                profile, requirements, match, evidence_report
            )
        except Exception as exc:
            # Graceful degradation: the deterministic match result stays; only
            # the narrative enrichment is lost.
            return {"errors": [f"generate_grounded_recommendation: {exc}"]}

        # Attach evidence to the match result even if narrative step degrades
        updated_match = match.model_copy(
            update={
                "summary": narrative.summary,
                "strengths": narrative.strengths,
                "recommendations": narrative.recommendations,
                "evidence": evidence_report,
            }
        )
        return {"match_result": updated_match}

    return generate_grounded_recommendation


def make_generate_resume_optimization_node(llm_service: LLMService):
    """Generate resume optimization suggestions constrained by evidence."""
    def generate_resume_optimization(state) -> dict:
        try:
            requirements = state["job_requirements"]
            match = state["match_result"]
            evidence_report = state.get("evidence_report")
            resume_evidence = state.get("resume_evidence")

            if evidence_report is None or not resume_evidence:
                return {"resume_optimization": None}

            optimizer = ResumeOptimizer(llm_service)
            result = optimizer.optimize(
                requirements, match, evidence_report, resume_evidence
            )
            return {"resume_optimization": result}
        except Exception as exc:
            return {
                "errors": [f"generate_resume_optimization: {exc}"],
                "resume_optimization": None,
            }

    return generate_resume_optimization


def make_validate_resume_optimization_node():
    """Validate generated optimization and drop invalid actions."""
    def validate_resume_optimization(state) -> dict:
        optimization = state.get("resume_optimization")
        evidence_report = state.get("evidence_report")
        resume_evidence = state.get("resume_evidence") or []
        match = state.get("match_result")

        if optimization is None or evidence_report is None or match is None:
            return {}

        try:
            validator = OptimizationValidator(resume_evidence)
            validated = validator.validate(optimization, evidence_report, match)
            return {"resume_optimization": validated}
        except Exception as exc:
            return {
                "errors": [f"validate_resume_optimization: {exc}"],
            }

    return validate_resume_optimization


def route_after_extraction(state) -> str:
    if state.get("candidate_profile") is None or state.get("job_requirements") is None:
        return "abort"
    return "proceed"


def route_after_match(state) -> str:
    if state.get("match_result") is None:
        return "abort"
    return "proceed"
