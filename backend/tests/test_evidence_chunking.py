"""Tests for deterministic resume chunking from raw text."""

from app.services.evidence.chunking import ResumeChunker


def _alex_resume_text():
    return """\
Alex Chen
AI Engineer
alex.chen@example.com | (555) 987-6543 | San Francisco, CA

Summary
AI/ML engineer with 5+ years building production LLM-powered applications, agentic workflows, and scalable backend services.

Skills
Python, LangChain, LangGraph, OpenAI API, FastAPI, Pydantic, Docker, AWS

Experience
Senior AI Engineer - AutoCorp (2021-2025)
- Designed and deployed agentic workflows using LangGraph and LangChain for customer support automation
- Built REST APIs with FastAPI and Pydantic, serving 1M+ requests/day

Software Engineer - DataMind (2019-2021)
- Developed Python microservices for data pipelines and model serving

Education
M.S. Computer Science - Tech University (2019)
B.S. Computer Science - State University (2017)
"""


def test_chunker_creates_summary_chunk_from_raw_text():
    text = "Summary\nExperienced software engineer."
    chunks = ResumeChunker().chunk(text)
    assert len(chunks) == 1
    assert chunks[0].id == "summary"
    assert chunks[0].section == "summary"
    assert chunks[0].text == "Experienced software engineer."


def test_chunker_creates_skills_chunk_from_raw_text():
    text = "Skills\nPython, FastAPI, Docker"
    chunks = ResumeChunker().chunk(text)
    assert len(chunks) == 1
    assert chunks[0].id == "skills"
    assert chunks[0].section == "skills"
    assert chunks[0].text == "Python, FastAPI, Docker"


def test_chunker_creates_experience_chunks_from_raw_text():
    text = """Experience
Senior Engineer - TechCorp (2020-2023)
- Built REST APIs
- Mentored team"""
    chunks = ResumeChunker().chunk(text)
    assert len(chunks) == 3
    assert chunks[0].id == "exp-0"
    assert chunks[0].section == "experience"
    assert chunks[0].text == "Senior Engineer - TechCorp (2020-2023)"
    assert chunks[1].id == "exp-0-b0"
    assert chunks[1].text == "Built REST APIs"
    assert chunks[2].id == "exp-0-b1"
    assert chunks[2].text == "Mentored team"


def test_chunker_creates_education_chunks_from_raw_text():
    text = """Education
B.S. Computer Science - State University (2017)
M.S. Computer Science - Tech University (2019)"""
    chunks = ResumeChunker().chunk(text)
    assert len(chunks) == 2
    assert chunks[0].id == "edu-0"
    assert chunks[0].section == "education"
    assert chunks[0].text == "B.S. Computer Science - State University (2017)"


def test_every_evidence_text_is_substring_of_raw_resume():
    text = _alex_resume_text()
    chunks = ResumeChunker().chunk(text)
    assert chunks
    for chunk in chunks:
        assert chunk.text in text, f"{chunk.id!r} not a substring: {chunk.text!r}"


def test_chunker_preserves_bullet_text_exactly():
    text = """Experience
Engineer - Corp (2020-2023)
-   Bullet with extra spaces   """
    chunks = ResumeChunker().chunk(text)
    bullet_chunk = next(c for c in chunks if c.id == "exp-0-b0")
    assert bullet_chunk.text in text


def test_chunker_metadata_indices():
    text = """Experience
Engineer - Corp (2020-2023)
- Bullet 1
- Bullet 2"""
    chunks = ResumeChunker().chunk(text)
    assert chunks[0].metadata["entry_index"] == "0"
    assert chunks[1].metadata["entry_index"] == "0"
    assert chunks[1].metadata["bullet_index"] == "0"
    assert chunks[2].metadata["bullet_index"] == "1"


def test_chunker_empty_resume():
    chunks = ResumeChunker().chunk("")
    assert chunks == []


def test_chunker_alex_chen_resume_has_expected_chunks():
    text = _alex_resume_text()
    chunks = ResumeChunker().chunk(text)
    ids = {c.id for c in chunks}
    assert "summary" in ids
    assert "skills" in ids
    assert "exp-0" in ids
    assert "exp-0-b0" in ids
    assert "exp-0-b1" in ids
    assert "exp-1" in ids
    assert "exp-1-b0" in ids
    assert "edu-0" in ids
    assert "edu-1" in ids


def _chinese_resume_text():
    return """\
张三
zhangsan@example.com | 13800000000

教育背景
示例大学｜信息技术硕士 2024.09–2026.09
示例理工大学｜机械设计制造及其自动化｜工学学士 2018.09–2023.06

专业能力
AI/LLM：RAG、LangChain、Embedding
数据与分析：Python、SQL、Pandas
开发与原型：React、Flask、Docker、Git、Linux

工作经历
示例显示科技有限公司｜自动化助理工程师 2023.06–2023.12
•就职于自动化部门，负责AGV、STK、OHT等自动化物流搬运设备的日常维护。
•参与新自动化设备安装、调试及上线导入。

示例机电设备有限公司｜机器人工程师 2022.06–2022.12
•负责工业搬运机器人的日常维护、故障排查及现场调试。
•参与新机器人设备安装、调试及产线导入。

项目经历
眼动数据分析平台｜示例大学Capstone
技术：Python/Pandas/React/Flask/Docker
•参与迭代面向眼动研究数据的Web分析平台。
•负责Visualisation模块，将结果划分为三类。
"""


def test_chunker_recognizes_chinese_section_headers():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    sections = {c.section for c in chunks}
    assert "education" in sections
    assert "skills" in sections
    assert "experience" in sections
    assert "projects" in sections


def test_chunker_produces_non_zero_chunks_for_chinese_resume():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    assert len(chunks) > 0


def test_chunker_creates_cjk_experience_entry_chunks():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    entries = [c for c in chunks if c.section == "experience" and c.id.startswith("exp-") and "-b" not in c.id]
    assert len(entries) == 2
    assert "示例显示科技有限公司" in entries[0].text
    assert "自动化助理工程师" in entries[0].text
    assert entries[0].metadata["title"] == "自动化助理工程师"
    assert entries[0].metadata["company"] == "示例显示科技有限公司"


def test_chunker_creates_cjk_bullet_chunks():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    bullets = [c for c in chunks if c.section == "experience" and "-b" in c.id]
    assert len(bullets) == 4
    assert any("AGV" in c.text for c in bullets)


def test_chunker_creates_project_chunks_for_chinese_resume():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    projects = [c for c in chunks if c.section == "projects"]
    assert len(projects) == 3  # entry + 2 bullets
    assert any(c.id == "proj-0" for c in projects)


def test_chinese_evidence_texts_are_substrings_of_raw_resume():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    assert chunks
    for chunk in chunks:
        assert chunk.text in text, f"{chunk.id!r} not a substring: {chunk.text!r}"


def test_chunker_does_not_treat_company_date_line_as_header():
    text = _chinese_resume_text()
    chunks = ResumeChunker().chunk(text)
    # If a company/date line were mis-detected as a header, chunk count would
    # be lower or sections would be empty.
    assert len(chunks) >= 10
    assert not any(c.section == "summary" for c in chunks)
