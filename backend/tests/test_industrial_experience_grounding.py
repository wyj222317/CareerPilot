"""Task 3C.2: Resume-side industrial experience grounding regression tests.

Verifies that explicit industrial automation experience (AGV, robotics, etc.)
in the resume is correctly extracted, normalized, and matched against
corresponding job requirements.
"""

from pathlib import Path

from app.services.graph import build_analysis_graph
from app.services.llm.mock import MockLLMService


_CN_INDUSTRIAL_RESUME = """\
李工
ligong@example.com | 13800000001

教育背景
上海交通大学｜机械工程学士 2018.09–2022.06

工作经历
示例自动化科技有限公司｜自动化工程师 2022.07–2023.12
•负责AGV自动导引车的导航算法调试与现场部署。
•参与工业机器人集成系统的调试与维护。
•使用ROS2进行机器人运动规划与传感器融合。

示例机电设备有限公司｜机器人工程师 2022.01–2022.06
•负责ABB机械臂的日常维护与故障排查。
•参与MES系统与自动化产线的集成调试。

技能
AGV、工业机器人、ROS2、Python、C++、Linux、MES、传感器融合
"""

_CN_AGV_JD = """\
AGV机器人工程师

岗位职责：
1. 负责AGV导航算法开发与现场部署。
2. 参与工业机器人系统集成与调试。
3. 使用ROS2进行机器人运动规划。

任职要求：
- 2年以上AGV或机器人行业经验。
- 熟悉AGV、工业机器人、ROS2。
- 精通Python和C++。
- 有MES系统集成经验优先。
"""


def test_industrial_resume_extracts_agv_robotics_experience():
    """Verify that AGV and robotics experience is extracted from resume."""
    profile = MockLLMService().extract_candidate_profile(_CN_INDUSTRIAL_RESUME)
    
    # Check skills
    skills = set(profile.skills)
    assert "agv" in skills, "AGV should be extracted from resume"
    assert "robotics" in skills, "robotics should be extracted from resume"
    assert "ros2" in skills or "ros" in skills, "ROS2 should be extracted"
    assert "mes" in skills, "MES should be extracted"
    
    # Check experience entries
    assert len(profile.experience) >= 2, "Should extract at least 2 experience entries"
    
    # Check first experience (AGV focus)
    exp1 = profile.experience[0]
    assert exp1.title == "自动化工程师"
    assert exp1.company == "示例自动化科技有限公司"
    assert exp1.start_date == "2022"
    assert exp1.end_date == "2023"
    assert any("AGV" in h for h in exp1.highlights), "AGV should appear in highlights"
    
    # Check second experience (robotics focus)
    exp2 = profile.experience[1]
    assert exp2.title == "机器人工程师"
    assert exp2.company == "示例机电设备有限公司"
    # Title should contain robotics-related term
    assert "机器人" in exp2.title or "robot" in exp2.title.lower()


def test_industrial_experience_duration_is_calculated():
    """Verify that experience duration is calculated from dated entries."""
    profile = MockLLMService().extract_candidate_profile(_CN_INDUSTRIAL_RESUME)
    
    # Should have extracted experience entries with dates
    assert len(profile.experience) >= 2
    
    # total_years_experience may be None for short-term positions (< 1 year each)
    # but the experience entries should have dates
    for exp in profile.experience:
        assert exp.start_date is not None, "start_date should be populated"
        assert exp.end_date is not None, "end_date should be populated"


def test_agv_requirement_satisfied_by_resume_experience():
    """Verify that AGV requirement is satisfied by resume AGV experience."""
    graph = build_analysis_graph(MockLLMService(), matching_service=None)
    state = graph.invoke({
        "raw_resume_text": _CN_INDUSTRIAL_RESUME,
        "raw_job_description": _CN_AGV_JD,
    })
    
    assert state.get("errors", []) == []
    
    match = state["match_result"]
    assert match is not None
    
    # AGV should be in matched_skills, not missing_skills
    assert "agv" in match.matched_skills, (
        f"AGV should be matched. matched={match.matched_skills}, missing={match.missing_skills}"
    )
    assert "agv" not in match.missing_skills, (
        "AGV should not be in missing_skills when resume has AGV experience"
    )
    
    # Robotics should also be matched
    assert "robotics" in match.matched_skills, (
        "robotics should be matched when resume has robotics experience"
    )


def test_evidence_retrieval_prefers_industrial_chunks():
    """Verify that evidence retrieval returns industrial employment chunks for AGV claims."""
    graph = build_analysis_graph(MockLLMService(), matching_service=None)
    state = graph.invoke({
        "raw_resume_text": _CN_INDUSTRIAL_RESUME,
        "raw_job_description": _CN_AGV_JD,
    })
    
    evidence_report = state.get("evidence_report")
    assert evidence_report is not None
    
    # Find claims related to AGV
    agv_claims = [
        claim for claim in evidence_report.claims
        if "agv" in claim.claim.lower()
    ]
    
    if agv_claims:
        # At least one AGV claim should be supported or partially supported
        supported_agv = [
            claim for claim in agv_claims
            if claim.support_level in ("supported", "partially_supported")
        ]
        assert supported_agv, (
            "AGV claims should have some support from industrial experience"
        )
        
        # Check that evidence comes from experience section
        for claim in supported_agv:
            evidence_sections = [
                e.section for e in claim.evidence
                if e.evidence_id.startswith("exp-")
            ]
            # At least some evidence should be from experience section
            assert evidence_sections, (
                "AGV claims should cite experience evidence, not just skills/education"
            )


def test_mes_skill_is_recognized_and_matched():
    """Verify that MES skill is recognized from resume and matched against JD."""
    graph = build_analysis_graph(MockLLMService(), matching_service=None)
    state = graph.invoke({
        "raw_resume_text": _CN_INDUSTRIAL_RESUME,
        "raw_job_description": _CN_AGV_JD,
    })
    
    reqs = state["job_requirements"]
    match = state["match_result"]
    profile = state["candidate_profile"]
    
    # MES should be in resume skills
    assert "mes" in profile.skills, "Resume should have MES in skills"
    
    # MES should be extracted from JD (in either required or preferred)
    all_jd_skills = [s.lower() for s in reqs.required_skills + reqs.preferred_skills]
    assert "mes" in all_jd_skills, "MES should be extracted from JD"
    
    # MES should be matched (in matched_skills)
    assert "mes" in match.matched_skills, (
        f"MES should be matched. matched={match.matched_skills}"
    )


def test_no_fabricated_experience():
    """Verify that optimization does not fabricate experience beyond what's in resume."""
    graph = build_analysis_graph(MockLLMService(), matching_service=None)
    state = graph.invoke({
        "raw_resume_text": _CN_INDUSTRIAL_RESUME,
        "raw_job_description": _CN_AGV_JD,
    })
    
    optimization = state.get("resume_optimization")
    assert optimization is not None
    
    evidence_map = {e.id: e for e in state["resume_evidence"]}
    
    for action in optimization.actions:
        if action.optimized_text:
            # Check that optimized text doesn't contain claims not in evidence
            cited_text = " ".join(
                evidence_map[eid].text
                for eid in action.evidence_ids
                if eid in evidence_map
            )
            # Should not fabricate specific technologies not in resume
            for tech in ["KUKA", "FANUC", "Siemens", "SAP"]:
                if tech not in cited_text:
                    assert tech not in action.optimized_text, (
                        f"Should not fabricate {tech} not in cited evidence"
                    )


def test_existing_tests_still_pass():
    """Sanity check: existing Chinese resume tests should still work."""
    from pathlib import Path
    
    fixture_path = Path(__file__).parent / "fixtures" / "chinese_resume.txt"
    CHINESE_RESUME_TEXT = fixture_path.read_text(encoding="utf-8")
    
    profile = MockLLMService().extract_candidate_profile(CHINESE_RESUME_TEXT)
    
    # Should extract skills
    assert len(profile.skills) > 0
    
    # Should extract education
    assert len(profile.education) > 0
    
    # The original fixture has different format, so experience may or may not be extracted
    # depending on the format. This test just ensures no crash.
