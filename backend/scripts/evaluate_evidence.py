"""Evaluation fixture for evidence retrieval.

Runs the evidence grounding pipeline on a fixed resume + JD and reports
retrieval precision against expected evidence.

Usage (from backend/):
    python scripts/evaluate_evidence.py

By default uses mock embeddings for determinism. Set EMBEDDING_PROVIDER=openai
to exercise the real provider (requires API key).
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.analysis import MatchAnalysisService
from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import get_embedding_service
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore
from app.services.llm.mock import MockLLMService
from app.services.matching import MatchingService
from app.services.skill_normalization import SkillNormalizer

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

# Expected evidence for key claims. Keys are substrings matched against claim text.
EXPECTED_EVIDENCE = {
    # Required skills (split from compound requirements by the matching pipeline)
    "python": {"expected_support_level": "supported"},
    "langchain": {"expected_support_level": "supported"},
    "langgraph": {"expected_support_level": "supported"},
    "openai": {"expected_support_level": "supported"},
    "deepseek": {"expected_support_level": "supported"},
    "qwen": {"expected_support_level": "supported"},
    "fastapi": {"expected_support_level": "supported"},
    "pydantic": {"expected_support_level": "supported"},
    "postgresql": {"expected_support_level": "supported"},
    "docker": {"expected_support_level": "supported"},
    "aws": {"expected_support_level": "supported"},
    # Responsibilities
    "Design and implement agentic workflows": {"expected_support_level": "supported"},
    "Integrate and benchmark multiple LLM providers": {"expected_support_level": "partially_supported"},
    "Build robust, type-safe REST APIs": {"expected_support_level": "partially_supported"},
    "Collaborate with frontend engineers on agent interfaces": {"expected_support_level": "supported"},
    "Mentor team members on LLM best practices": {"expected_support_level": "partially_supported"},
}


def _format_skills(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "(none)"


def _semantics_label(group) -> str:
    if group is None:
        return "N/A (not a skill claim)"
    return f"{'OR' if group.kind == 'any' else 'AND'} ({group.kind})"


def _claim_diagnostics(claim, normalizer: SkillNormalizer) -> None:
    """Print detailed diagnostics for a single claim."""
    group = normalizer.parse_claim(claim.claim)

    print(f"Claim: {claim.claim}")
    print(f"  Type: {claim.claim_type}")
    print(f"  Atomic skills: {_format_skills(normalizer.extract_canonical_skills(claim.claim))}")
    print(f"  Semantics: {_semantics_label(group)}")
    print(f"  Final support level: {claim.support_level}")

    if claim.evidence:
        print("  Retrieved evidence:")
        for ev in claim.evidence:
            lexical = normalizer.extract_canonical_skills(ev.text)
            print(f"    {ev.evidence_id}  score={ev.score:.3f}  section={ev.section}")
            print(f"      Lexical skills: {_format_skills(lexical)}")
            snippet = ev.text.replace("\n", " ")
            print(f"      Text: {snippet[:120]}{'...' if len(snippet) > 120 else ''}")
    else:
        print("  Retrieved evidence: (none)")


def main() -> int:
    print("=" * 80)
    print("Evidence Retrieval Evaluation Fixture")
    print("=" * 80)
    print()

    # Run analysis pipeline
    llm = MockLLMService()
    embedder = get_embedding_service()
    chunker = ResumeChunker()
    retriever = EvidenceRetriever(embedder=embedder)
    normalizer = SkillNormalizer()

    service = MatchAnalysisService(
        llm_service=llm,
        matching_service=MatchingService(),
        embedding_service=embedder,
        vector_store_factory=InMemoryVectorStore,
        chunker=chunker,
        retriever=retriever,
    )

    result = service.analyze(RESUME_TEXT, JOB_DESCRIPTION)

    print(f"Overall score: {result.match.overall_score}/100")
    print(f"Matched skills: {', '.join(result.match.matched_skills)}")
    print()

    # Check evidence report
    if result.match.evidence is None:
        print("ERROR: No evidence report generated!")
        return 1

    report = result.match.evidence
    print(f"Evidence report: {report.supported_count} supported, "
          f"{report.partially_supported_count} partial, "
          f"{report.unsupported_count} unsupported")
    print()

    # Per-claim diagnostics
    print("-" * 80)
    print("Per-Claim Diagnostics")
    print("-" * 80)
    print()

    hits = 0
    total_expected = 0
    unmatched_expected = []

    for claim in report.claims:
        _claim_diagnostics(claim, normalizer)

        # Find expected spec, if any, using substring matching.
        matched_key = next(
            (key for key in EXPECTED_EVIDENCE if key.lower() in claim.claim.lower()),
            None,
        )
        if matched_key is not None:
            total_expected += 1
            expected = EXPECTED_EVIDENCE[matched_key]
            expected_level = expected["expected_support_level"]
            passed = claim.support_level == expected_level

            # Optional secondary checks: if the fixture specifies an expected
            # section or keywords, the retrieved evidence must satisfy them.
            expected_section = expected.get("expected_section")
            expected_keywords = expected.get("expected_keywords", [])
            if expected_section:
                passed = passed and any(
                    e.section == expected_section for e in claim.evidence
                )
            if expected_keywords:
                passed = passed and any(
                    any(kw.lower() in e.text.lower() for kw in expected_keywords)
                    for e in claim.evidence
                )

            status = "PASS" if passed else "FAIL"
            print(f"  Expected support level: {expected_level}")
            print(f"  Pass/Fail: {status}")
            if passed:
                hits += 1
        else:
            print("  Expected support level: N/A (not in fixture spec)")
            print("  Pass/Fail: N/A")

        print()

    # Claims in EXPECTED_EVIDENCE not found in the report
    for key in EXPECTED_EVIDENCE:
        found = any(key.lower() in c.claim.lower() for c in report.claims)
        if not found:
            unmatched_expected.append(key)
            total_expected += 1

    if unmatched_expected:
        print("Missing expected claims (not generated by pipeline):")
        for key in unmatched_expected:
            print(f"  - {key}")
        print()

    # Summary metrics
    print("-" * 80)
    print("Summary Metrics")
    print("-" * 80)
    print(f"Expected claims evaluated: {total_expected}")
    print(f"Correct support level + evidence location: {hits}")
    precision = hits / total_expected if total_expected > 0 else 0
    print(f"Precision: {hits}/{total_expected} = {precision:.0%}")
    print("-" * 80)

    if precision < 0.6:
        print("\nWARNING: Retrieval precision below 60%")
        return 1

    print("\n✓ Evaluation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
