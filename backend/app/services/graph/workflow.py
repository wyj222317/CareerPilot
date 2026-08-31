from langgraph.graph import END, START, StateGraph

from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import EmbeddingService, get_embedding_service
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore, VectorStore
from app.services.graph.nodes import (
    make_analyze_job_description_node,
    make_build_evidence_index_node,
    make_calculate_match_node,
    make_generate_grounded_recommendation_node,
    make_generate_resume_optimization_node,
    make_parse_resume_node,
    make_retrieve_evidence_node,
    make_validate_resume_optimization_node,
    route_after_extraction,
    route_after_match,
)
from app.services.graph.state import AnalysisState
from app.services.llm.base import LLMService
from app.services.matching import MatchingService


def build_analysis_graph(
    llm_service: LLMService,
    matching_service: MatchingService | None = None,
    embedding_service: EmbeddingService | None = None,
    vector_store_factory=None,
    chunker: ResumeChunker | None = None,
    retriever: EvidenceRetriever | None = None,
):
    """Compile the analysis workflow.

    parse_resume → analyze_job_description → calculate_match →
    build_evidence_index → retrieve_evidence →
    generate_grounded_recommendation, with conditional abort edges after
    extraction and after matching. Evidence nodes are best-effort.
    """
    matching = matching_service or MatchingService()
    embedder = embedding_service or get_embedding_service()
    store_factory = vector_store_factory or InMemoryVectorStore
    chunk = chunker or ResumeChunker()
    retrieve = retriever or EvidenceRetriever(embedder=embedder)

    builder = StateGraph(AnalysisState)
    builder.add_node("parse_resume", make_parse_resume_node(llm_service))
    builder.add_node("analyze_job_description", make_analyze_job_description_node(llm_service))
    builder.add_node("calculate_match", make_calculate_match_node(matching))
    builder.add_node(
        "build_evidence_index",
        make_build_evidence_index_node(chunk, embedder, store_factory),
    )
    builder.add_node("retrieve_evidence", make_retrieve_evidence_node(retrieve))
    builder.add_node(
        "generate_grounded_recommendation",
        make_generate_grounded_recommendation_node(llm_service),
    )
    builder.add_node(
        "generate_resume_optimization",
        make_generate_resume_optimization_node(llm_service),
    )
    builder.add_node(
        "validate_resume_optimization",
        make_validate_resume_optimization_node(),
    )

    builder.add_edge(START, "parse_resume")
    builder.add_edge("parse_resume", "analyze_job_description")
    builder.add_conditional_edges(
        "analyze_job_description",
        route_after_extraction,
        {"proceed": "calculate_match", "abort": END},
    )
    builder.add_conditional_edges(
        "calculate_match",
        route_after_match,
        {"proceed": "build_evidence_index", "abort": END},
    )
    builder.add_edge("build_evidence_index", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "generate_grounded_recommendation")
    builder.add_edge("generate_grounded_recommendation", "generate_resume_optimization")
    builder.add_edge("generate_resume_optimization", "validate_resume_optimization")
    builder.add_edge("validate_resume_optimization", END)
    return builder.compile()
