"""Evaluation fixture for evidence-constrained resume optimization.

Runs the full pipeline on a fixed resume + JD and prints every optimization
action, gap warning, and validation issue. The fixture demonstrates that
unsupported facts (skills, platforms, metrics, years) are blocked.

Usage (from backend/):
    python scripts/evaluate_optimization.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import (
    EvidenceReport,
    OptimizationAction,
    ResumeEvidence,
    ResumeOptimizationResult,
)
from app.services.analysis import MatchAnalysisService
from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import get_embedding_service
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore
from app.services.llm.mock import MockLLMService
from app.services.matching import MatchingService
from app.services.optimization.validator import OptimizationValidator

RESUME_TEXT = """\
Alex Chen
AI Engineer
alex.chen@example.com | (555) 987-6543 | San Francisco, CA

Summary
AI/ML engineer with 5+ years building production LLM-powered applications, agentic workflows, and scalable backend services. Experienced with LangGraph, LangChain, OpenAI/DeepSeek/Qwen APIs, FastAPI, and cloud deployment.

Skills
Python, LangChain, LangGraph, OpenAI API, DeepSeek API, FastAPI, Pydantic, SQL, PostgreSQL, Docker, Kubernetes, AWS, React, TypeScript, Git, CI/CD

Experience
Senior AI Engineer - AutoCorp (2021-2025)
- Designed and deployed agentic workflows using LangGraph and LangChain for customer support automation
- Integrated OpenAI, DeepSeek, and Qwen APIs via a unified provider abstraction with structured JSON output
- Built REST APIs with FastAPI and Pydantic, serving 1M+ requests/day
- Mentored junior engineers on LLM application design and prompt engineering

Software Engineer - DataMind (2019-2021)
- Developed Python microservices for data pipelines and model serving
- Deployed containerized services with Docker and Kubernetes on AWS
- Collaborated with product teams to ship React/TypeScript frontend features

Education
M.S. Computer Science - Tech University (2019)
B.S. Computer Science - State University (2017)
"""

JOB_DESCRIPTION = """\
AI/Agent Engineer

About the role
We are building the next generation of autonomous AI agents for enterprise operations. You will design, build, and ship agentic systems that reason, plan, and execute complex tasks.

Required qualifications
- 4+ years of professional software engineering experience
- Expert-level Python
- Production experience with LangGraph or LangChain
- Hands-on experience integrating LLM APIs (OpenAI, DeepSeek, or Qwen)
- Strong API design skills with FastAPI and Pydantic
- Comfortable with PostgreSQL and relational data modeling
- Experience deploying services with Docker and AWS

Preferred qualifications
- Experience with React and TypeScript
- Familiarity with vector databases (Pinecone, Weaviate, or Chroma)
- Knowledge of CI/CD pipelines and infrastructure as code

Responsibilities
- Design and implement agentic workflows and reasoning loops
- Integrate and benchmark multiple LLM providers
- Build robust, type-safe REST APIs for agent orchestration
- Collaborate with frontend engineers on agent interfaces
- Mentor team members on LLM best practices
"""


def main() -> int:
    print("=" * 80)
    print("Evidence-Constrained Resume Optimization Evaluation Fixture")
    print("=" * 80)
    print()

    llm = MockLLMService()
    embedder = get_embedding_service()
    chunker = ResumeChunker()
    retriever = EvidenceRetriever(embedder=embedder)

    service = MatchAnalysisService(
        llm_service=llm,
        matching_service=MatchingService(),
        embedding_service=embedder,
        vector_store_factory=InMemoryVectorStore,
        chunker=chunker,
        retriever=retriever,
    )

    result = service.analyze(RESUME_TEXT, JOB_DESCRIPTION)
    optimization = result.optimization

    print(f"Overall score: {result.match.overall_score}/100")
    print(f"Matched skills: {', '.join(result.match.matched_skills)}")
    print()

    if optimization is None:
        print("ERROR: No optimization result generated!")
        return 1

    print(f"Optimization summary: {optimization.summary}")
    print(
        f"Actions: {len(optimization.actions)} · "
        f"Gap warnings: {len(optimization.gap_warnings)} · "
        f"Validation issues: {len(optimization.validation_issues)}"
    )
    print()

    # Build evidence lookup for display.
    evidence_by_id = {
        match.evidence_id: match
        for claim in (result.match.evidence.claims if result.match.evidence else [])
        for match in claim.evidence
    }

    print("-" * 80)
    print("Optimization Actions")
    print("-" * 80)
    print()

    for action in optimization.actions:
        print(f"Section: {action.target_section}")
        print(f"Action type: {action.action_type}")
        print(f"Targets: {', '.join(action.target_requirements)}")
        print(f"Evidence IDs: {', '.join(action.evidence_ids)}")
        print(f"Original: {action.original_text}")
        print(f"Optimized: {action.optimized_text}")
        print(f"Rationale: {action.rationale}")
        print("Cited evidence:")
        for eid in action.evidence_ids:
            ev_match = evidence_by_id.get(eid)
            if ev_match:
                snippet = ev_match.text[:120]
                suffix = "..." if len(ev_match.text) > 120 else ""
                print(f"  [{ev_match.section}] {snippet}{suffix}")
            else:
                print(f"  [{eid}] (not found in evidence report)")
        print()

    if optimization.gap_warnings:
        print("-" * 80)
        print("Gap Warnings")
        print("-" * 80)
        print()
        for warning in optimization.gap_warnings:
            print(f"Requirement: {warning.requirement}")
            print(f"Rationale: {warning.rationale}")
            print()

    if optimization.validation_issues:
        print("-" * 80)
        print("Validation Issues")
        print("-" * 80)
        print()
        for issue in optimization.validation_issues:
            print(f"Field: {issue.field}")
            print(f"Message: {issue.message}")
            print()

    print("-" * 80)
    print("Adversarial Checks")
    print("-" * 80)
    print()

    import re

    _YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

    def _years(text: str) -> set[str]:
        return set(_YEAR_RE.findall(text))

    checks = [
        ("AWS fabrication blocked", any(
            "aws" not in (a.optimized_text or "").lower()
            for a in optimization.actions
        )),
        ("Kubernetes fabrication blocked", all(
            "kubernetes" not in (a.optimized_text or "").lower()
            for a in optimization.actions
            if "kubernetes" not in (a.original_text or "").lower()
        )),
        ("No invented metrics", all(
            "%" not in (a.optimized_text or "")
            for a in optimization.actions
        )),
        ("No invented years", all(
            _years(a.optimized_text or "").issubset(
                {
                    year
                    for eid in a.evidence_ids
                    for year in _years(
                        (evidence_by_id.get(eid).text or "") if evidence_by_id.get(eid) else ""
                    )
                }
            )
            for a in optimization.actions
            if a.target_section == "experience"
        )),
        ("All rewrites cite evidence", all(
            a.evidence_ids for a in optimization.actions
        )),
    ]

    passed = 0
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}")
        if ok:
            passed += 1

    print()
    print(f"Adversarial checks: {passed}/{len(checks)} passed")
    print("-" * 80)

    if passed < len(checks):
        print("\nWARNING: Some adversarial checks failed")
        return 1

    print()
    print("-" * 80)
    print("Adversarial structural validation")
    print("-" * 80)
    print()

    # Simulate an LLM that returns a surface_existing_evidence action without
    # optimized_text. The validator must reject it and treat it as invalid.
    adversarial_evidence = [
        ResumeEvidence(
            id="exp-0",
            text="Built REST APIs with FastAPI.",
            section="experience",
            source="test",
        )
    ]
    adversarial_result = ResumeOptimizationResult(
        actions=[
            OptimizationAction(
                action_type="surface_existing_evidence",
                target_section="experience",
                original_text="Built REST APIs with FastAPI.",
                optimized_text=None,
                rationale="Surface the cited evidence.",
                target_requirements=["FastAPI"],
                evidence_ids=["exp-0"],
            )
        ],
        summary="adversarial",
    )
    validator = OptimizationValidator(adversarial_evidence)
    validated = validator.validate(
        adversarial_result,
        result.match.evidence or EvidenceReport(claims=[]),
        result.match,
    )

    if validated.actions or not any(
        "optimized_text" in issue.field for issue in validated.validation_issues
    ):
        print("[FAIL] surface_existing_evidence with null optimized_text was accepted")
        return 1

    print(
        "[PASS] surface_existing_evidence with null optimized_text rejected: "
        f"{validated.validation_issues[0].message}"
    )

    print("\n✓ Evaluation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
