"""Deterministic resume chunking into evidence units.

Splits raw resume text into ResumeEvidence chunks by section. Every chunk's
`text` is an exact substring of the original resume text so claims can be
traced back to the original source, not an LLM-transformed representation.
"""

import re

from app.models.schemas import CandidateProfile, ResumeEvidence
from app.services.text_normalization import normalize_header

# Section headers we recognize. Values are normalized header tokens.
_SECTION_HEADERS = {
    "summary": {
        "summary",
        "profile",
        "objective",
        "professional summary",
        # Chinese
        "个人简介",
        "自我评价",
        "个人总结",
        "简介",
        "求职概况",
    },
    "skills": {
        "skills",
        "technical skills",
        "core competencies",
        "key skills",
        # Chinese
        "专业能力",
        "专业技能",
        "技能",
        "技术能力",
        "核心技能",
        "it技能",
        "技术栈",
    },
    "experience": {
        "experience",
        "work experience",
        "employment history",
        "professional experience",
        "career history",
        # Chinese
        "工作经历",
        "工作经验",
        "实习经历",
        "职业经历",
    },
    "projects": {
        "projects",
        "project experience",
        # Chinese
        "项目经历",
        "项目经验",
    },
    "education": {
        "education",
        "academic background",
        "qualifications",
        # Chinese
        "教育背景",
        "教育经历",
        "学历",
    },
}

# English entry line: "Software Engineer - Example Corp (2020 - 2022)".
_EXPERIENCE_LINE_RE = re.compile(
    r"^(?P<title>[^(]{2,60}?)\s+-\s+(?P<company>[^(]{2,60}?)\s+"
    r"\((?P<start>\d{4})(?:\s*-\s*(?P<end>\d{4}|[Pp]resent))?\)\s*$"
)

# CJK entry line: "公司｜职位 2023.06–2023.12" or "公司 | 职位 2023.06-2023.12".
_CJK_ENTRY_RE = re.compile(
    r"^(?P<company>[^|｜丨]{2,60}?)\s*[|｜丨]\s*"
    r"(?P<title>[^\d(]{2,60}?)\s+"
    r"\(?(?P<start>(?:19|20)\d{2})\s*[-./年]\s*(?P<start_m>1[0-2]|0?[1-9])\s*月?"
    r"\s*(?:[-–—~～－]|[至到])\s*"
    r"(?P<end>(?:19|20)\d{2})\s*[-./年]\s*(?P<end_m>1[0-2]|0?[1-9])\s*月?\)?\s*$"
)

# Project entries often omit dates, e.g. "眼动数据分析平台｜示例大学Capstone".
_CJK_PROJECT_ENTRY_RE = re.compile(
    r"^(?P<title>[^|｜丨]{2,60}?)\s*[|｜丨]\s*(?P<company>[^|｜丨]{2,60}?)\s*$"
)

_BULLET_PREFIX = ("-", "•")
_CJK_BULLET_PREFIX = ("·", "▪", "‣", "–", "—", "*")

# A line is probably not a header if it looks like content.
_MAX_HEADER_LEN = 30
_HEADER_DIGIT_RE = re.compile(r"\d")


def _detect_section(line: str) -> str | None:
    """Return section key if the line is a recognized section header.

    Headers are normalized and matched against a curated set. Lines that are
    too long or contain digits are rejected to avoid treating company/date
    lines as headers.
    """
    normalized = normalize_header(line)
    if not normalized or len(normalized) > _MAX_HEADER_LEN:
        return None
    if _HEADER_DIGIT_RE.search(normalized):
        return None
    for section, headers in _SECTION_HEADERS.items():
        if normalized in headers:
            return section
    return None


def _is_bullet(line: str) -> bool:
    """Return True if the line starts with a recognized bullet marker."""
    stripped = line.lstrip()
    if stripped.startswith(_BULLET_PREFIX):
        return True
    if stripped and stripped[0] in _CJK_BULLET_PREFIX:
        return True
    # CJK bullet with no space, e.g. "•就职于..."
    if stripped.startswith("•"):
        return True
    return False


def _strip_bullet(line: str) -> str:
    """Remove bullet prefix and surrounding whitespace."""
    stripped = line.lstrip()
    if stripped.startswith(_BULLET_PREFIX):
        # Strip the first bullet char and any following whitespace.
        return stripped[1:].lstrip()
    if stripped.startswith("•"):
        return stripped[1:].lstrip()
    if stripped and stripped[0] in _CJK_BULLET_PREFIX:
        return stripped[1:].lstrip()
    return stripped


class ResumeChunker:
    """Deterministically chunk raw resume text into evidence units.

    The returned ResumeEvidence.text values are exact substrings of the input
    resume text (modulo harmless whitespace stripping of bullet prefixes).
    """

    def chunk(
        self,
        raw_resume_text: str,
        profile: CandidateProfile | None = None,
    ) -> list[ResumeEvidence]:
        lines = raw_resume_text.splitlines()
        chunks: list[ResumeEvidence] = []

        current_section: str | None = None
        section_buffer: list[str] = []

        entry_index = 0
        bullet_index = 0
        current_title = ""
        current_company = ""

        def flush_summary_or_skills():
            nonlocal section_buffer
            if not section_buffer:
                return
            text = "\n".join(section_buffer).strip()
            if not text:
                section_buffer = []
                return
            if current_section == "summary":
                chunks.append(
                    ResumeEvidence(
                        id="summary",
                        text=text,
                        section="summary",
                        source="Summary",
                        metadata={},
                    )
                )
            elif current_section == "skills":
                chunks.append(
                    ResumeEvidence(
                        id="skills",
                        text=text,
                        section="skills",
                        source="Skills",
                        metadata={"count": str(len(text.split(",")))},
                    )
                )
            section_buffer = []

        for raw_line in lines:
            stripped = raw_line.strip()
            detected = _detect_section(stripped)
            if detected:
                flush_summary_or_skills()
                current_section = detected
                entry_index = 0
                bullet_index = 0
                current_title = ""
                current_company = ""
                continue

            if not stripped:
                continue

            if current_section == "summary":
                section_buffer.append(stripped)
            elif current_section == "skills":
                section_buffer.append(stripped)
            elif current_section == "experience":
                match = _EXPERIENCE_LINE_RE.match(stripped) or _CJK_ENTRY_RE.match(stripped)
                if match:
                    bullet_index = 0
                    current_title = match.group("title").strip()
                    current_company = match.group("company").strip()
                    chunks.append(
                        ResumeEvidence(
                            id=f"exp-{entry_index}",
                            text=stripped,
                            section="experience",
                            source=f"Experience: {current_title} @ {current_company}",
                            metadata={
                                "entry_index": str(entry_index),
                                "title": current_title,
                                "company": current_company,
                            },
                        )
                    )
                    entry_index += 1
                elif _is_bullet(stripped):
                    bullet_text = _strip_bullet(stripped)
                    entry_idx = entry_index - 1
                    chunks.append(
                        ResumeEvidence(
                            id=f"exp-{entry_idx}-b{bullet_index}",
                            text=bullet_text,
                            section="experience",
                            source=(
                                f"Experience: {current_title} @ {current_company} "
                                f"(bullet {bullet_index + 1})"
                            ).strip(),
                            metadata={
                                "entry_index": str(entry_idx),
                                "bullet_index": str(bullet_index),
                                "title": current_title,
                                "company": current_company,
                            },
                        )
                    )
                    bullet_index += 1
            elif current_section == "projects":
                match = (
                    _EXPERIENCE_LINE_RE.match(stripped)
                    or _CJK_ENTRY_RE.match(stripped)
                    or _CJK_PROJECT_ENTRY_RE.match(stripped)
                )
                if match:
                    bullet_index = 0
                    current_title = match.group("title").strip()
                    current_company = match.group("company").strip()
                    chunks.append(
                        ResumeEvidence(
                            id=f"proj-{entry_index}",
                            text=stripped,
                            section="projects",
                            source=f"Project: {current_title}",
                            metadata={
                                "entry_index": str(entry_index),
                                "title": current_title,
                                "company": current_company,
                            },
                        )
                    )
                    entry_index += 1
                elif _is_bullet(stripped):
                    bullet_text = _strip_bullet(stripped)
                    entry_idx = entry_index - 1
                    chunks.append(
                        ResumeEvidence(
                            id=f"proj-{entry_idx}-b{bullet_index}",
                            text=bullet_text,
                            section="projects",
                            source=(
                                f"Project: {current_title} "
                                f"(bullet {bullet_index + 1})"
                            ).strip(),
                            metadata={
                                "entry_index": str(entry_idx),
                                "bullet_index": str(bullet_index),
                                "title": current_title,
                                "company": current_company,
                            },
                        )
                    )
                    bullet_index += 1
            elif current_section == "education":
                chunks.append(
                    ResumeEvidence(
                        id=f"edu-{entry_index}",
                        text=stripped,
                        section="education",
                        source=f"Education: {stripped}",
                        metadata={"entry_index": str(entry_index)},
                    )
                )
                entry_index += 1

        flush_summary_or_skills()
        return chunks
