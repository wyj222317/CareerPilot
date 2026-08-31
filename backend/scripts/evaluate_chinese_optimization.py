"""Chinese resume + JD optimization fixture using the real LLM provider.

This fixture verifies that the Phase 4 optimizer pipeline handles a non-English
resume and job description end-to-end and that structurally invalid actions are
still caught by OptimizationValidator.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.analysis import MatchAnalysisService
from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import get_embedding_service
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore
from app.services.graph import build_analysis_graph
from app.services.llm import get_llm_service
from app.services.matching import MatchingService

RESUME_TEXT = """\
张伟
自动化工程师
zhangwei@example.com | 138-0000-0000

Summary
工业自动化专家，熟悉PLC、MES、AGV、STK、OHT设备维护，具备Linux系统操作和Python脚本经验。

Skills
Linux, Python, PLC, MES, AGV, STK, OHT, 工业自动化

Experience
自动化工程师 - 示例工厂 (2022-2024)
- 维护PLC和MES设备
- 调试AGV、STK、OHT机械
- 使用Python编写设备状态监控脚本
- 与运营团队协作排查软硬件问题

Education
本科 - 电子信息工程 - 示例大学 (2022)
"""

JOB_DESCRIPTION = """\
机器人/AI数据采集工程师

岗位要求：
- 熟悉Linux系统和网络配置
- 掌握C/Python/Shell编程
- 能独立排查软硬件问题
- 有机器人/AI行业经验
- 有VR/动作捕捉/机器人数据采集经验
- 熟悉摄像头/雷达/IMU等机器人传感器
- 了解485/CAN/HTTP/WebRTC/ROS2等通信协议
- 较强的执行力、团队合作和问题解决能力
- 了解数据隐私法规
- 具备合规安全数据采集意识
- 2年以上工作经验

岗位职责：
- 为机器人数据采集实验室配置Linux系统和网络
- 编写C、Python和Shell脚本控制机器人并处理传感器数据
- 独立排查数据采集过程中的软硬件问题
- 使用VR、动作捕捉和摄像头/雷达/IMU等传感器
- 使用485、CAN、HTTP、WebRTC、ROS2等通信协议
- 遵守数据隐私法规和合规安全的数据采集实践
- 与机器人/AI团队协作交付高质量数据集

学历要求：
- 计算机科学、机器人、AI或相关专业本科及以上学历
"""


def main() -> int:
    print("=" * 80)
    print("Chinese Resume + JD Optimization Fixture")
    print("=" * 80)
    print()

    llm = get_llm_service()
    embedder = get_embedding_service()
    chunker = ResumeChunker()
    retriever = EvidenceRetriever(embedder=embedder)

    graph = build_analysis_graph(
        llm_service=llm,
        matching_service=MatchingService(),
        embedding_service=embedder,
        vector_store_factory=InMemoryVectorStore,
        chunker=chunker,
        retriever=retriever,
    )

    state = graph.invoke(
        {
            "raw_resume_text": RESUME_TEXT,
            "raw_job_description": JOB_DESCRIPTION,
        }
    )

    errors = state.get("errors", [])
    if errors:
        print("Pipeline errors:")
        for err in errors:
            print(f"  - {err}")
        print()

    profile = state.get("candidate_profile")
    requirements = state.get("job_requirements")
    match = state.get("match_result")
    evidence_report = state.get("evidence_report")
    resume_evidence = state.get("resume_evidence", [])
    optimization = state.get("resume_optimization")

    print(f"Profile: {profile.name if profile else None}")
    print(f"Requirements title: {requirements.title if requirements else None}")
    print(f"Resume evidence chunks: {len(resume_evidence)}")
    print(
        f"Evidence report: "
        f"{evidence_report.supported_count if evidence_report else 0} supported, "
        f"{evidence_report.partially_supported_count if evidence_report else 0} partial, "
        f"{evidence_report.unsupported_count if evidence_report else 0} unsupported"
    )
    print(f"Overall score: {match.overall_score if match else None}/100")
    print(f"Matched skills: {', '.join(match.matched_skills) if match else ''}")
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

    if optimization.validation_issues:
        print("-" * 80)
        print("Validation Issues (caught and dropped by validator)")
        print("-" * 80)
        for issue in optimization.validation_issues:
            print(f"Field: {issue.field}")
            print(f"Message: {issue.message}")
            print()

    surviving_issues = [
        issue
        for issue in optimization.validation_issues
        if issue.field not in ("unsupported_concept", "cross_scope_evidence")
    ]
    if surviving_issues:
        print("ERROR: Non-factual validation issues remain on surviving actions.")
        return 1

    print(
        f"✓ Chinese optimization fixture passed: "
        f"{len(optimization.actions)} clean action(s), "
        f"{len(optimization.validation_issues)} factual violation(s) caught and dropped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
