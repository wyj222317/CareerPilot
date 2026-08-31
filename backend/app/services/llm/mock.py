import re
from datetime import date

from app.models.schemas import (
    CandidateProfile,
    EducationItem,
    EvidenceReport,
    ExperienceItem,
    GapWarning,
    JobRequirements,
    MatchResult,
    OptimizationAction,
    RecommendationNarrative,
    ResumeEvidence,
    ResumeOptimizationResult,
    RewrittenBullet,
)
from app.services.llm.base import LLMService
from app.services.skill_normalization import SkillNormalizer
from app.services.text_normalization import build_term_pattern, contains_cjk

# English-only fallback for contexts where the full normalizer isn't needed.
WORD_SKILLS = {
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "django", "flask", "fastapi", "pydantic", "spring", "sql", "mysql", "postgresql",
    "mongodb", "redis", "graphql", "docker", "kubernetes", "terraform",
    "aws", "azure", "gcp", "git", "linux", "bash", "html", "css", "sass",
    "tailwind", "jest", "pytest", "selenium", "playwright", "react native",
    "machine learning", "deep learning", "data science", "data engineering",
    "nlp", "computer vision", "pytorch", "tensorflow", "scikit-learn",
    "pandas", "numpy", "spark", "hadoop", "airflow", "kafka", "tableau",
    "power bi", "excel", "agile", "scrum", "kanban", "microservices",
    "devops", "kotlin", "swift", "scala", "rust",
    # LLM / agent stack
    "langchain", "langgraph", "openai", "deepseek", "qwen",
}

SYMBOL_SKILLS = {"c++", "c#", "node.js", "next.js", ".net", "ci/cd"}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
# English-style years mention: "5+ years", "3 years", etc.
_YEARS_RE = re.compile(r"(\d+)\s*\+?\s*years", re.IGNORECASE)
# Chinese-style years expressions: "3年以上", "5年经验", "至少3年".
_CN_YEARS_RE = re.compile(r"(?:至少)?(\d+)\s*年(?:以上|经验|工作)?")

# Section-header markers that split required vs preferred skills.
_PREFERRED_MARKERS = (
    "preferred", "nice to have", "bonus", "plus:",
    "优先", "加分",
)

# Education line detection (English + Chinese).
_DEGREE_KEYWORDS = (
    "b.s.", "b.a.", "m.s.", "m.a.", "ph.d.", "mba", "bachelor", "master",
    "本科", "硕士", "博士", "学士", "学历", "学位",
)
_CN_DEGREE_SUFFIXES = ("本科", "硕士", "博士", "学士", "学历", "学位")

# English experience line: "Title - Company (2018-2023)" or "(2018-Present)".
_EXP_LINE_RE = re.compile(
    r"^(?P<title>[^(]{2,60}?)\s*-\s*(?P<company>[^(]{2,60}?)\s+"
    r"\((?P<start>\d{4})(?:\s*-\s*(?P<end>\d{4}|[Pp]resent))?\)\s*$"
)
# Chinese experience line: "职位 - 公司（2018-至今）" / "（2018-现在）".
_CN_EXP_LINE_RE = re.compile(
    r"^(?P<title>[^（(]{2,60}?)\s*[-—–－]\s*(?P<company>[^（(]{2,60}?)\s*"
    r"[（(](?P<start>\d{4})(?:\s*[-—–－至到]\s*(?P<end>\d{4}|至今|现在|今))?[）)]\s*$"
)
# Chinese experience line (alternate format): "公司｜职位 YYYY.MM–YYYY.MM"
_CN_EXP_LINE_ALT_RE = re.compile(
    r"^(?P<company>[^｜|]{2,60}?)\s*[｜|]\s*(?P<title>[^０-９0-9]{2,60}?)\s+"
    r"(?P<start>\d{4})[.\/\-](?P<start_month>\d{1,2})\s*[–\-—]\s*"
    r"(?P<end>\d{4}|至今|现在|今)(?:[.\/\-](?P<end_month>\d{1,2}))?\s*$"
)
_TRAILING_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")

# Bullet prefixes used across resume/JD styles.
_BULLET_PREFIX = ("-", "•", "·", "▪", "‣")
_CN_ENUMERATION_RE = re.compile(r"^(?:\d+|[一二三四五六七八九十]+)\s*[.、．]\s*")
_CN_PAREN_ENUMERATION_RE = re.compile(r"^[（(]\s*(?:\d+|[一二三四五六七八九十]+)\s*[）)]\s*")

# Chinese responsibility / requirement section headers.
_CN_RESPONSIBILITY_HEADERS = (
    "岗位职责", "工作内容", "工作职责", "职责描述", "主要职责",
    "任职要求", "岗位要求", "职位要求", "任职资格",
)
_EDUCATION_SECTION_HEADERS = ("education", "学历", "教育背景", "教育经历")


def find_skills(text: str) -> list[str]:
    """Return canonical skills mentioned in *text*, bilingual and CJK-aware.

    Uses :class:`SkillNormalizer` for the full alias table (English + Chinese)
    with correct boundary handling for both ASCII (``\\b``) and CJK terms.
    """
    if not text:
        return []
    return SkillNormalizer.extract_canonical_skills(text)


def _is_education_line(line: str) -> bool:
    lowered = line.lower()
    if any(keyword in lowered for keyword in _DEGREE_KEYWORDS):
        return True
    return any(suffix in line for suffix in _CN_DEGREE_SUFFIXES)


def _strip_bullet(line: str) -> str:
    for prefix in _BULLET_PREFIX:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    new_line, n_subs = _CN_ENUMERATION_RE.subn("", line, count=1)
    if n_subs:
        return new_line.strip()
    new_line, n_subs = _CN_PAREN_ENUMERATION_RE.subn("", line, count=1)
    if n_subs:
        return new_line.strip()
    return line.strip()


def _parse_education(line: str) -> EducationItem:
    year = None
    year_match = _TRAILING_YEAR_RE.search(line)
    if year_match:
        year = year_match.group(1)
        line = line[: year_match.start()].rstrip()

    # Helper: given two parts, identify which is the degree (the one containing
    # a degree keyword) and which is the institution.
    def _split_pair(left: str, right: str) -> EducationItem | None:
        for keyword in _DEGREE_KEYWORDS:
            if keyword in left.lower():
                return EducationItem(degree=left.strip(), institution=right.strip() or None, year=year)
            if keyword in right.lower():
                return EducationItem(degree=right.strip(), institution=left.strip() or None, year=year)
        return None

    # Explicit "degree - institution" separator.
    if " - " in line:
        left, right = (part.strip() for part in line.split(" - ", 1))
        return _split_pair(left, right) or EducationItem(degree=left, institution=right, year=year)

    # Chinese resumes often use "|" / "｜" as the field separator:
    #   "清华大学｜本科 2024" / "示例理工大学｜机械设计制造及其自动化｜工学学士 2018"
    for sep in ("｜", "|"):
        if sep in line:
            parts = [part.strip() for part in line.split(sep)]
            # Trailing year may sit on the last segment without parentheses:
            # "工学学士 2018.09–2023.06". Strip a trailing 4-digit year group.
            right_year_match = re.match(
                r"^(.*?)[\s,，]+(\d{4}(?:\.\d{2})?(?:[–—\-~]\S*)?)\s*$", parts[-1]
            )
            if right_year_match and year is None:
                year = right_year_match.group(2)
                parts[-1] = right_year_match.group(1).strip()
            # Pick the segment containing a degree keyword as the degree;
            # join the rest as institution.
            degree_idx = None
            for idx, part in enumerate(parts):
                if any(kw in part.lower() for kw in _DEGREE_KEYWORDS):
                    degree_idx = idx
                    break
            if degree_idx is not None:
                degree = parts[degree_idx]
                institution = " ".join(p for i, p in enumerate(parts) if i != degree_idx)
                return EducationItem(
                    degree=degree, institution=institution or None, year=year
                )
            # No keyword matched — treat left as institution, last as degree.
            return EducationItem(
                degree=parts[-1], institution=" ".join(parts[:-1]) or None, year=year
            )

    # Chinese education often ends with the degree keyword:
    #   "清华大学 本科" / "北京大学计算机硕士学位"
    for suffix in _CN_DEGREE_SUFFIXES:
        if line.endswith(suffix):
            idx = len(line) - len(suffix)
            institution = line[:idx].rstrip() or None
            return EducationItem(degree=suffix, institution=institution, year=year)

    # Plain English: "B.S. Computer Science" — whole line is the degree.
    return EducationItem(degree=line, institution=None, year=year)


def _is_responsibility_header(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith("responsibilit"):
        return True
    return any(header in line for header in _CN_RESPONSIBILITY_HEADERS)


def _is_education_section_header(line: str) -> bool:
    lowered = line.lower()
    return any(header in lowered for header in _EDUCATION_SECTION_HEADERS)


def _parse_experience_years(experience: list[ExperienceItem]) -> int | None:
    total = 0
    for item in experience:
        if not (item.start_date and item.end_date and item.start_date.isdigit()):
            continue
        if item.end_date in ("Present", "present", "至今", "现在", "今"):
            end = date.today().year
        elif item.end_date.isdigit():
            end = int(item.end_date)
        else:
            continue
        total += max(0, end - int(item.start_date))
    return total or None


class MockLLMService(LLMService):
    """Deterministic offline stand-in for a real LLM provider.

    Uses simple heuristics (skill lexicon, line-pattern regexes) so the full
    pipeline runs and is testable without any API keys. Bilingual: handles
    both English and Chinese resumes / job descriptions.
    """

    def extract_candidate_profile(self, resume_text: str) -> CandidateProfile:
        lines = [line.strip() for line in resume_text.splitlines()]

        name = None
        for line in lines:
            if not line:
                continue
            if len(line) <= 60 and "@" not in line and not any(ch.isdigit() for ch in line):
                name = line
            break

        email_match = _EMAIL_RE.search(resume_text)
        phone_match = _PHONE_RE.search(resume_text)

        experience: list[ExperienceItem] = []
        education: list[EducationItem] = []
        open_entry: ExperienceItem | None = None
        in_education_section = False
        for line in lines:
            if not line:
                open_entry = None
                in_education_section = False
                continue
            if _is_education_section_header(line):
                in_education_section = True
                open_entry = None
                continue
            if line.startswith(_BULLET_PREFIX) or _CN_ENUMERATION_RE.match(line) or _CN_PAREN_ENUMERATION_RE.match(line):
                if open_entry is not None:
                    open_entry.highlights.append(_strip_bullet(line))
                continue
            if _is_education_line(line) or (in_education_section and not _EXP_LINE_RE.match(line) and not _CN_EXP_LINE_RE.match(line) and not _CN_EXP_LINE_ALT_RE.match(line)):
                education.append(_parse_education(line))
                open_entry = None
                continue

            entry = None
            end_date_raw: str | None = None
            en_match = _EXP_LINE_RE.match(line)
            cn_match = _CN_EXP_LINE_RE.match(line)
            cn_alt_match = _CN_EXP_LINE_ALT_RE.match(line)
            if en_match and "@" not in line:
                entry = ExperienceItem(
                    title=en_match.group("title").strip(),
                    company=en_match.group("company").strip(),
                    start_date=en_match.group("start"),
                    end_date=en_match.group("end"),
                )
                end_date_raw = en_match.group("end")
            elif cn_match and "@" not in line:
                entry = ExperienceItem(
                    title=cn_match.group("title").strip(),
                    company=cn_match.group("company").strip(),
                    start_date=cn_match.group("start"),
                    end_date=cn_match.group("end"),
                )
                end_date_raw = cn_match.group("end")
            elif cn_alt_match and "@" not in line:
                entry = ExperienceItem(
                    title=cn_alt_match.group("title").strip(),
                    company=cn_alt_match.group("company").strip(),
                    start_date=cn_alt_match.group("start"),
                    end_date=cn_alt_match.group("end"),
                )
                end_date_raw = cn_alt_match.group("end")

            if entry is not None:
                if end_date_raw in ("至今", "现在", "今"):
                    entry.end_date = "Present"
                experience.append(entry)
                open_entry = entry
                continue
            open_entry = None
            in_education_section = False

        years_match = _YEARS_RE.search(resume_text) or _CN_YEARS_RE.search(resume_text)
        total_years = (
            int(years_match.group(1))
            if years_match
            else _parse_experience_years(experience)
        )

        return CandidateProfile(
            name=name,
            email=email_match.group(0) if email_match else None,
            phone=phone_match.group(0).strip() if phone_match else None,
            skills=find_skills(resume_text),
            total_years_experience=total_years,
            experience=experience,
            education=education,
        )

    def extract_job_requirements(self, job_description: str) -> JobRequirements:
        lowered = job_description.lower()
        split_at = min(
            (idx for idx in (lowered.find(marker) for marker in _PREFERRED_MARKERS) if idx != -1),
            default=-1,
        )
        if split_at == -1:
            required, preferred = find_skills(job_description), []
        else:
            required = find_skills(job_description[:split_at])
            preferred = [
                skill for skill in find_skills(job_description[split_at:]) if skill not in required
            ]
        years_match = _YEARS_RE.search(job_description) or _CN_YEARS_RE.search(job_description)

        responsibilities: list[str] = []
        education_requirements: list[str] = []
        in_responsibilities = False
        in_education = False
        for line in job_description.splitlines():
            line = line.strip()
            if not line:
                in_responsibilities = False
                in_education = False
                continue
            if _is_responsibility_header(line):
                in_responsibilities = True
                in_education = False
                continue
            if _is_education_section_header(line):
                in_education = True
                in_responsibilities = False
                continue
            if in_responsibilities:
                if (
                    line.startswith(_BULLET_PREFIX)
                    or _CN_ENUMERATION_RE.match(line)
                    or _CN_PAREN_ENUMERATION_RE.match(line)
                ):
                    responsibilities.append(_strip_bullet(line))
                    continue
                in_responsibilities = False
            if in_education:
                if (
                    line.startswith(_BULLET_PREFIX)
                    or _CN_ENUMERATION_RE.match(line)
                    or _CN_PAREN_ENUMERATION_RE.match(line)
                ):
                    education_requirements.append(_strip_bullet(line).rstrip("."))
                    continue
                in_education = False
            if _is_education_line(line):
                education_requirements.append(_strip_bullet(line).rstrip("."))

        lines = [line.strip() for line in job_description.splitlines() if line.strip()]
        title = lines[0] if lines and len(lines[0]) <= 80 else None
        return JobRequirements(
            title=title,
            required_skills=required,
            preferred_skills=preferred,
            min_years_experience=int(years_match.group(1)) if years_match else None,
            responsibilities=responsibilities,
            education_requirements=education_requirements,
        )

    def generate_recommendation(
        self, profile: CandidateProfile, requirements: JobRequirements, match: MatchResult
    ) -> RecommendationNarrative:
        if requirements.required_skills:
            summary = (
                f"You match {len(match.matched_skills)} of "
                f"{len(requirements.required_skills)} required skills, for an overall fit of "
                f"{match.overall_score}/100."
            )
        else:
            summary = f"Overall fit: {match.overall_score}/100."
        if match.missing_skills:
            summary += f" Closing the {len(match.missing_skills)} identified skill gap(s) would raise your score."

        strengths = list(match.strengths)
        if profile.education:
            top = profile.education[0]
            line = f"Education: {top.degree}"
            if top.institution:
                line += f" from {top.institution}"
            strengths.append(line)

        recommendations = list(match.recommendations)
        if match.missing_skills:
            recommendations.insert(0, f"Priority gap to close: {match.missing_skills[0].skill}.")

        return RecommendationNarrative(
            summary=summary,
            strengths=strengths,
            recommendations=recommendations,
        )

    def generate_grounded_recommendation(
        self,
        profile: CandidateProfile,
        requirements: JobRequirements,
        match: MatchResult,
        evidence_report: EvidenceReport | None = None,
    ) -> RecommendationNarrative:
        narrative = self.generate_recommendation(profile, requirements, match)
        if evidence_report:
            summary = (
                f"{narrative.summary} "
                f"Evidence: {evidence_report.supported_count} supported, "
                f"{evidence_report.partially_supported_count} partial, "
                f"{evidence_report.unsupported_count} unsupported."
            )
            return RecommendationNarrative(
                summary=summary,
                strengths=narrative.strengths,
                recommendations=narrative.recommendations,
            )
        return narrative

    def generate_resume_optimization(
        self,
        requirements: JobRequirements,
        match: MatchResult,
        evidence_report: EvidenceReport,
        policy_actions: list[OptimizationAction],
        gap_warnings: list[GapWarning],
        relevant_evidence: list[ResumeEvidence],
    ) -> ResumeOptimizationResult:
        """Deterministic, evidence-constrained resume optimization.

        The mock rewrites only what is directly supported by the cited evidence.
        It never invents skills, metrics, employers, or dates.
        """
        evidence_by_id = {e.id: e for e in relevant_evidence}
        actions: list[OptimizationAction] = []
        rewritten_bullets: list[RewrittenBullet] = []

        for action in policy_actions:
            if action.action_type == "gap_warning":
                continue

            evidence_ids = [
                eid for eid in action.evidence_ids if eid in evidence_by_id
            ]
            if not evidence_ids:
                continue

            original_text = evidence_by_id[evidence_ids[0]].text
            required_skills = set(find_skills(" ".join(action.target_requirements)))
            cited_skills = set(
                skill
                for eid in evidence_ids
                for skill in find_skills(evidence_by_id[eid].text)
            )

            supported_skills = required_skills & cited_skills
            unsupported_skills = required_skills - cited_skills

            if supported_skills:
                # Rewrite to foreground supported skills while keeping the
                # original factual statement intact. Never mention unsupported
                # skills in the optimized bullet.
                optimized = (
                    f"{original_text} — aligns with "
                    f"{', '.join(sorted(supported_skills))} requirement."
                )
            else:
                # No skill overlap: surface the existing evidence unchanged.
                optimized = original_text

            new_action = OptimizationAction(
                action_type="rewrite",
                target_section=action.target_section,
                original_text=original_text,
                optimized_text=optimized,
                rationale=action.rationale,
                target_requirements=action.target_requirements,
                evidence_ids=evidence_ids,
                confidence=action.confidence,
            )
            actions.append(new_action)
            rewritten_bullets.append(
                RewrittenBullet(
                    section=action.target_section,
                    original=original_text,
                    rewritten=optimized,
                    evidence_ids=evidence_ids,
                    target_requirements=action.target_requirements,
                )
            )

        summary = (
            f"Generated {len(actions)} evidence-constrained rewrite(s) and "
            f"{len(gap_warnings)} gap warning(s)."
        )

        return ResumeOptimizationResult(
            actions=actions,
            rewritten_bullets=rewritten_bullets,
            gap_warnings=list(gap_warnings),
            validation_issues=[],
            summary=summary,
        )
