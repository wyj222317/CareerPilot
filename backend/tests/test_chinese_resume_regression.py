"""End-to-end regression test for the Chinese resume hardening."""

from pathlib import Path

from app.models.schemas import (
    CandidateProfile,
    EducationItem,
    ExperienceItem,
    JobRequirements,
    RecommendationNarrative,
    ResumeOptimizationResult,
)
from app.services.evidence.chunking import ResumeChunker
from app.services.graph import build_analysis_graph
from app.services.llm.base import LLMService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "chinese_resume.txt"


def _chinese_resume_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def _chinese_job_text() -> str:
    return """\
Job Responsibilities
Responsible for the full lifecycle implementation of robot data engineering, including data collection, cleaning, review, annotation, archiving, and uploading.
Responsible for analyzing and resolving hardware anomalies of robots, network communication failures, and data acquisition software/system issues encountered during the data collection process.
Responsible for formulating and implementing data collection standards, quality assessment criteria, training manuals for data collectors, and data version control methods.

Job Requirements
Full-time undergraduate degree or higher from a regular enrollment program, with majors in computer science, automation, mechanical engineering, or related fields.
Proficient in Linux systems and network configuration, with C++/Python/Shell programming skills, capable of independently troubleshooting software and hardware issues.
More than 2 years of experience in the AI or robotics industry, with priority given to candidates familiar with VR, motion capture (tracking), or robotic data collection solutions.
Prior experience with common robot sensors (such as cameras, radars, IMUs, etc.) and communication protocols (e.g., 485, CAN, HTTP, WebRTC, ROS2) is preferred.
Understand data privacy regulations (such as the Personal Information Protection Law) and possess compliance awareness in data collection as well as secure operational standards.
"""


class ChineseStubLLM(LLMService):
    """Deterministic stub that returns the expected Chinese structures."""

    def extract_candidate_profile(self, resume_text: str) -> CandidateProfile:
        return CandidateProfile(
            name="张三",
            email="zhangsan@example.com",
            phone="13800000000",
            skills=[
                "RAG",
                "LangChain",
                "Embedding",
                "PromptEngineering",
                "Python",
                "SQL",
                "Pandas",
                "数据预处理",
                "React",
                "Flask",
                "JavaScript",
                "Docker",
                "Git",
                "Linux",
                "MES",
                "AGV/STK/OHT",
                "工业机器人",
                "PLC基础",
            ],
            experience=[
                ExperienceItem(
                    title="自动化助理工程师",
                    company="示例显示科技有限公司",
                    start_date="2023.06",
                    end_date="2023.12",
                    highlights=[
                        "就职于自动化部门，负责AGV、STK、OHT等自动化物流搬运设备的日常维护。",
                        "参与新自动化设备安装、调试及上线导入。",
                        "接触MES（制造执行系统）及自动化生产业务流程。",
                    ],
                ),
                ExperienceItem(
                    title="机器人工程师",
                    company="示例机电设备有限公司",
                    start_date="2022.06",
                    end_date="2022.12",
                    highlights=[
                        "负责工业搬运机器人的日常维护、故障排查及现场调试。",
                        "参与新机器人设备安装、调试及产线导入；接触PLC编程及工业自动化控制系统。",
                    ],
                ),
            ],
            education=[
                EducationItem(
                    degree="信息技术硕士",
                    institution="示例大学",
                    year="2024.09–2026.09",
                ),
                EducationItem(
                    degree="工学学士",
                    institution="示例理工大学",
                    year="2018.09–2023.06",
                ),
            ],
        )

    def extract_job_requirements(self, job_description: str) -> JobRequirements:
        return JobRequirements(
            title="Robotics / AI Data Collection Engineer",
            required_skills=[
                "Proficient in Linux systems and network configuration",
                "C++/Python/Shell programming skills",
                "Able to independently troubleshoot software and hardware issues",
                "More than 2 years of experience in the AI or robotics industry",
                "Understand data privacy regulations (such as the Personal Information Protection Law)",
                "Compliance awareness in data collection as well as secure operational standards",
            ],
            preferred_skills=[
                "Familiar with VR, motion capture (tracking), or robotic data collection solutions",
                "Experience with common robot sensors (such as cameras, radars, IMUs, etc.)",
                "Experience with communication protocols (e.g., 485, CAN, HTTP, WebRTC, ROS2)",
            ],
            min_years_experience=2,
            responsibilities=[
                "Responsible for the full lifecycle implementation of robot data engineering, including data collection, cleaning, review, annotation, archiving, and uploading.",
                "Responsible for analyzing and resolving hardware anomalies of robots, network communication failures, and data acquisition software/system issues encountered during the data collection process.",
                "Responsible for formulating and implementing data collection standards, quality assessment criteria, training manuals for data collectors, and data version control methods.",
            ],
            education_requirements=[
                "Full-time undergraduate degree or higher from a regular enrollment program, with majors in computer science, automation, mechanical engineering, or related fields."
            ],
        )

    def generate_recommendation(self, profile, requirements, match):
        return RecommendationNarrative(
            summary="Stub summary.",
            strengths=["Stub strength"],
            recommendations=["Stub recommendation"],
        )

    def generate_resume_optimization(
        self, requirements, match, evidence_report, policy_actions, gap_warnings, relevant_evidence
    ):
        # Return one grounded rewrite action per gap so the result is non-trivial
        # but still deterministic and validation-safe.
        actions = []
        for ev in relevant_evidence:
            if ev.section in ("experience", "projects"):
                actions.append(
                    {
                        "action_type": "surface_existing_evidence",
                        "target_section": ev.section,
                        "original_text": ev.text,
                        "optimized_text": ev.text,
                        "rationale": "Surface existing evidence that addresses the requirement.",
                        "target_requirements": ["robot data collection"],
                        "evidence_ids": [ev.id],
                        "confidence": 0.9,
                    }
                )
                if len(actions) >= 3:
                    break
        return ResumeOptimizationResult(
            actions=actions,
            rewritten_bullets=[],
            gap_warnings=[
                {"requirement": g.requirement, "rationale": g.rationale}
                for g in gap_warnings
            ],
            validation_issues=[],
            summary="Stub optimization for Chinese resume.",
        )


def test_chinese_resume_produces_non_zero_evidence_chunks():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    assert len(chunks) > 0
    sections = {c.section for c in chunks}
    assert sections == {"education", "skills", "experience", "projects"}


def test_chinese_resume_evidence_texts_are_substrings():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    for chunk in chunks:
        assert chunk.text in text


def test_chinese_pipeline_produces_evidence_report_and_optimization():
    graph = build_analysis_graph(ChineseStubLLM(), matching_service=None)
    state = graph.invoke(
        {
            "raw_resume_text": _chinese_resume_text(),
            "raw_job_description": _chinese_job_text(),
        }
    )

    assert state.get("errors", []) == []
    assert state["candidate_profile"] is not None
    assert state["job_requirements"] is not None
    assert state["match_result"] is not None

    chunks = state.get("resume_evidence", [])
    assert len(chunks) > 0

    evidence_report = state.get("evidence_report")
    assert evidence_report is not None
    assert len(evidence_report.claims) > 0

    optimization = state.get("resume_optimization")
    assert optimization is not None
    assert len(optimization.actions) > 0
    assert optimization.validation_issues == []


def test_chinese_pipeline_education_is_normalized():
    graph = build_analysis_graph(ChineseStubLLM(), matching_service=None)
    state = graph.invoke(
        {
            "raw_resume_text": _chinese_resume_text(),
            "raw_job_description": _chinese_job_text(),
        }
    )
    match = state["match_result"]
    education = next(c for c in match.score_breakdown if c.name == "education_fit")
    assert education.score == 100


def test_chinese_pipeline_experience_is_calculated_from_dates():
    graph = build_analysis_graph(ChineseStubLLM(), matching_service=None)
    state = graph.invoke(
        {
            "raw_resume_text": _chinese_resume_text(),
            "raw_job_description": _chinese_job_text(),
        }
    )
    match = state["match_result"]
    experience = next(c for c in match.score_breakdown if c.name == "experience_fit")
    assert experience.score == 50
    assert "calculated from resume dates" in experience.detail


def test_chinese_optimization_does_not_invent_unsupported_claims():
    graph = build_analysis_graph(ChineseStubLLM(), matching_service=None)
    state = graph.invoke(
        {
            "raw_resume_text": _chinese_resume_text(),
            "raw_job_description": _chinese_job_text(),
        }
    )
    optimization = state["resume_optimization"]
    evidence_map = {e.id: e for e in state["resume_evidence"]}
    for action in optimization.actions:
        for eid in action.evidence_ids:
            assert eid in evidence_map
        if action.optimized_text:
            # No ROS2/CAN/sensor claims should be invented beyond cited evidence.
            cited = " ".join(evidence_map[eid].text for eid in action.evidence_ids)
            for claim in ["ROS2", "CAN", "传感器"]:
                if claim not in cited:
                    assert claim not in action.optimized_text
