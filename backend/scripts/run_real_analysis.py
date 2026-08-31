"""Run a real end-to-end analysis using the configured LLM provider.

This script bypasses the HTTP server and calls MatchAnalysisService directly,
so it is the fastest way to exercise the full LangGraph pipeline with a real
provider such as DeepSeek.

Usage (from backend/):

    export LLM_PROVIDER=deepseek
    export LLM_API_KEY=your-api-key-here
    # optional overrides:
    # export LLM_MODEL=deepseek-chat
    # export LLM_BASE_URL=https://api.deepseek.com
    python scripts/run_real_analysis.py

The script reads the resume from sample_data/ai_agent_engineer_resume.pdf and
uses a realistic AI/Agent Engineer job description.
"""
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.analysis import MatchAnalysisService
from app.services.llm import get_llm_service
from app.services.pdf_extraction import extract_text_from_pdf

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
    provider = os.environ.get("LLM_PROVIDER", "mock")
    if provider != "mock" and not os.environ.get("LLM_API_KEY"):
        print(f"LLM_API_KEY must be set for provider={provider}", file=sys.stderr)
        return 1

    resume_path = BACKEND_DIR / "sample_data" / "ai_agent_engineer_resume.pdf"
    if not resume_path.exists():
        print(f"Resume not found: {resume_path}", file=sys.stderr)
        return 1

    with open(resume_path, "rb") as f:
        resume_text, page_count = extract_text_from_pdf(f.read())
    print(f"Extracted {len(resume_text)} characters from {page_count} page(s).\n")

    service = MatchAnalysisService(get_llm_service())
    result = service.analyze(resume_text, JOB_DESCRIPTION)

    print(json.dumps(result.model_dump(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
