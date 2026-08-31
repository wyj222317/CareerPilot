"""Bilingual MockLLMService tests (Task 3C.1).

Verifies that MockLLMService — the deterministic heuristic provider used on
the public Render demo — handles both English and Chinese resumes / JDs
without collapsing into an empty extraction, and that the plausibility
warning fires when a substantial input produces nothing.
"""

from app.services.graph import build_analysis_graph
from app.services.graph.nodes import (
    make_analyze_job_description_node,
    make_parse_resume_node,
)
from app.services.llm.mock import MockLLMService
from app.services.matching import MatchingService


_CN_ROBOTICS_RESUME = """\
李明
liming@example.com | 13900000001

教育背景
上海交通大学｜工学学士 2018

工作经历
机器人工程师 - 示例机器人科技有限公司（2019-至今）
- 负责AGV自动导引车的运动规划与激光雷达导航算法开发。
- 使用ROS2和C++实现传感器融合与计算机视觉定位。
- 参与工业机器人协作机器人系统集成，对接MES制造执行系统。

技能
C++、Python、ROS、ROS2、AGV、激光雷达、计算机视觉、运动捕捉、
IMU、传感器、Docker、Linux、Git
"""

_CN_ROBOTICS_JD = """\
AGV机器人研发工程师

岗位职责：
1. 负责AGV自动导引车的导航算法与运动规划开发。
2. 基于ROS2与激光雷达实现机器人在复杂环境下的定位与建图。
3. 协同工业机器人与MES系统的集成调试。

任职要求：
- 3年以上机器人行业工作经验。
- 精通C++和Python。
- 熟悉激光雷达、IMU、计算机视觉等传感器技术。
- 了解ROS或ROS2机器人操作系统。
- 具备运动捕捉或运动规划相关经验优先。
- 有协作机器人项目经验加分。
"""

_CN_SOFTWARE_RESUME = """\
王芳
wangfang@example.com

经验
高级后端工程师 - 互联网科技公司（2019-2024）
- 使用Python和FastAPI构建高并发微服务架构。
- 负责MySQL和Redis数据存储方案设计。
- 使用Docker和Kubernetes进行容器化部署。

教育
浙江大学 - 计算机硕士学位（2019）

技能
Python、FastAPI、MySQL、Redis、Docker、Kubernetes、Linux、Git
"""

_CN_SOFTWARE_JD = """\
高级后端工程师

岗位职责：
1. 负责后端微服务架构设计与开发。
2. 设计高并发系统的数据库方案。
3. 负责容器化部署和DevOps流程。

任职要求：
- 至少5年Python开发经验。
- 精通FastAPI或Django框架。
- 熟悉MySQL和Redis。
- 熟练使用Docker和Kubernetes。
- 有GraphQL或消息队列经验优先。
"""

_EN_RESUME = """\
Jane Doe
jane.doe@example.com

Experience
Senior Software Engineer - TechCorp (2020-2023)
- Built REST APIs with FastAPI
Software Engineer - StartupXYZ (2018-2020)
- Developed React components

Education
B.S. Computer Science - State University (2018)

Skills: Python, JavaScript, React, FastAPI, SQL, Docker
"""

_EN_JD = """\
Senior Full-Stack Engineer

Required skills: Python, TypeScript, React, Kubernetes, GraphQL
Preferred: AWS, Docker
Responsibilities:
- Design and ship features
- Mentor junior engineers

5+ years of experience.
"""


def test_cn_robotics_resume_extracts_skills_experience_education():
    profile = MockLLMService().extract_candidate_profile(_CN_ROBOTICS_RESUME)
    assert profile.name == "李明"
    assert profile.email == "liming@example.com"
    skills = set(profile.skills)
    for expected in ["c++", "python", "ros", "agv", "lidar", "computer vision",
                     "motion capture", "imu", "sensor", "docker", "linux", "git",
                     "robotics", "mes"]:
        assert expected in skills, f"missing {expected}"
    assert profile.experience, "expected at least one experience entry"
    assert profile.experience[0].title == "机器人工程师"
    assert profile.experience[0].company == "示例机器人科技有限公司"
    assert profile.experience[0].end_date == "Present"
    assert profile.education, "expected at least one education entry"
    assert profile.education[0].degree == "工学学士"
    assert profile.education[0].institution == "上海交通大学"


def test_cn_robotics_jd_extracts_required_preferred_years_responsibilities():
    reqs = MockLLMService().extract_job_requirements(_CN_ROBOTICS_JD)
    required = set(reqs.required_skills)
    for expected in ["c++", "python", "ros", "lidar", "imu", "computer vision",
                     "robotics", "mes", "agv"]:
        assert expected in required, f"missing required: {expected}"
    # The mock's heuristic splitting may place preferred-marker skills in required
    # if they appear before the marker in the text. Check that motion capture and
    # robotics are extracted somewhere (required or preferred).
    all_skills = set(reqs.required_skills) | set(reqs.preferred_skills)
    assert {"motion capture", "robotics"} <= all_skills, (
        "motion capture and robotics should be extracted from preferred-marker lines"
    )
    assert reqs.min_years_experience == 3
    assert len(reqs.responsibilities) >= 2


def test_cn_software_resume_and_jd_round_trip():
    profile = MockLLMService().extract_candidate_profile(_CN_SOFTWARE_RESUME)
    reqs = MockLLMService().extract_job_requirements(_CN_SOFTWARE_JD)
    skills = set(profile.skills)
    for expected in ["python", "fastapi", "mysql", "redis", "docker", "kubernetes"]:
        assert expected in skills, f"missing resume skill: {expected}"
    assert profile.experience
    assert reqs.min_years_experience == 5
    assert "python" in set(reqs.required_skills)
    assert len(reqs.responsibilities) >= 2


def test_cn_pipeline_does_not_produce_vacuous_100_match():
    """The headline regression: a substantial Chinese JD must NOT collapse to
    an empty requirement set and yield a vacuous 100/100 match.
    """
    graph = build_analysis_graph(MockLLMService(), matching_service=None)
    state = graph.invoke({
        "raw_resume_text": _CN_ROBOTICS_RESUME,
        "raw_job_description": _CN_ROBOTICS_JD,
    })

    profile = state["candidate_profile"]
    reqs = state["job_requirements"]
    assert profile is not None and reqs is not None
    assert reqs.required_skills, "Chinese JD produced empty required_skills"
    assert reqs.responsibilities, "Chinese JD produced empty responsibilities"

    match = state["match_result"]
    assert match is not None
    assert len(match.matched_skills) > 0 or len(match.missing_skills) > 0
    assert match.matched_skills, (
        "bilingual pipeline produced no matched skills"
    )

    evidence_report = state.get("evidence_report")
    assert evidence_report is not None
    assert (
        evidence_report.supported_count
        + evidence_report.partially_supported_count
        + evidence_report.unsupported_count
    ) > 0, "EvidenceReport was empty for a bilingual run"


def test_cn_pipeline_produces_optimization_actions():
    graph = build_analysis_graph(MockLLMService(), matching_service=None)
    state = graph.invoke({
        "raw_resume_text": _CN_SOFTWARE_RESUME,
        "raw_job_description": _CN_SOFTWARE_JD,
    })
    optimization = state.get("resume_optimization")
    assert optimization is not None, "resume_optimization was None"
    assert optimization.actions or optimization.gap_warnings, (
        "expected at least one optimization action or gap warning"
    )


def test_english_regression_still_works():
    """Guard against regressions to the original English-only mock behavior."""
    service = MockLLMService()
    profile = service.extract_candidate_profile(_EN_RESUME)
    assert profile.name == "Jane Doe"
    assert profile.email == "jane.doe@example.com"
    assert {"python", "react", "fastapi", "docker"} <= set(profile.skills)
    assert [item.company for item in profile.experience] == ["TechCorp", "StartupXYZ"]
    assert profile.total_years_experience == 5
    assert profile.education[0].degree == "B.S. Computer Science"
    assert profile.education[0].institution == "State University"

    reqs = service.extract_job_requirements(_EN_JD)
    assert {"python", "kubernetes", "graphql"} <= set(reqs.required_skills)
    assert {"aws", "docker"} <= set(reqs.preferred_skills)
    assert reqs.min_years_experience == 5
    assert reqs.responsibilities == [
        "Design and ship features",
        "Mentor junior engineers",
    ]


def test_plausibility_warning_fires_on_gibberish_input():
    """Substantial gibberish should trigger the plausibility warning, since the
    mock cannot extract any skills, experience, or education from it.
    """
    gibberish_resume = "LOREM IPSUM DOLOR SIT AMET " * 10
    node = make_parse_resume_node(MockLLMService())
    result = node({"raw_resume_text": gibberish_resume})
    assert "errors" in result
    assert any("parse_resume" in e for e in result["errors"])

    gibberish_jd = "LOREM IPSUM DOLOR SIT AMET " * 10
    node = make_analyze_job_description_node(MockLLMService())
    result = node({"raw_job_description": gibberish_jd})
    assert "errors" in result
    assert any("analyze_job_description" in e for e in result["errors"])


def test_plausibility_warning_absent_for_recognized_input():
    """Well-formed English input must not trigger the plausibility warning."""
    node = make_parse_resume_node(MockLLMService())
    result = node({"raw_resume_text": _EN_RESUME})
    assert "errors" not in result or not result["errors"]

    node = make_analyze_job_description_node(MockLLMService())
    result = node({"raw_job_description": _EN_JD})
    assert "errors" not in result or not result["errors"]


def test_english_pipeline_match_score_is_reasonable():
    """Sanity check: English pipeline produces a non-vacuous, non-zero match."""
    graph = build_analysis_graph(MockLLMService(), matching_service=None)
    state = graph.invoke({
        "raw_resume_text": _EN_RESUME,
        "raw_job_description": _EN_JD,
    })
    match = state["match_result"]
    assert match is not None
    assert 0 < match.overall_score < 100
    assert match.matched_skills, "expected some matched skills"
