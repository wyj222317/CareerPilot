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
_YEARS_RE = re.compile(r"(\d+)\s*\+?\s*years", re.IGNORECASE)
_PREFERRED_MARKERS = ("preferred", "nice to have", "bonus", "plus:")

_DEGREE_KEYWORDS = (
    "b.s.", "b.a.", "m.s.", "m.a.", "ph.d.", "mba", "bachelor", "master",
)
_EXP_LINE_RE = re.compile(
    r"^(?P<title>[^(]{2,60}?)\s+-\s+(?P<company>[^(]{2,60}?)\s+"
    r"\((?P<start>\d{4})(?:\s*-\s*(?P<end>\d{4}|[Pp]resent))?\)\s*$"
)
_TRAILING_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
_BULLET_PREFIX = ("-", "•")


def find_skills(text: str) -> list[str]:
    lowered = text.lower()
    found = {skill for skill in WORD_SKILLS if re.search(rf"\b{skill}\b", lowered)}
    found.update(skill for skill in SYMBOL_SKILLS if skill in lowered)
    return sorted(found, key=str.lower)


def _is_education_line(line: str) -> bool:
    lowered = line.lower()
    return any(keyword in lowered for keyword in _DEGREE_KEYWORDS)


def _strip_bullet(line: str) -> str:
    return line.lstrip("-• ").strip()


def _parse_education(line: str) -> EducationItem:
    year = None
    year_match = _TRAILING_YEAR_RE.search(line)
    if year_match:
        year = year_match.group(1)
        line = line[: year_match.start()].rstrip()
    degree, institution = line, None
    if " - " in line:
        degree, institution = (part.strip() for part in line.split(" - ", 1))
    return EducationItem(degree=degree, institution=institution, year=year)


def _parse_experience_years(experience: list[ExperienceItem]) -> int | None:
    total = 0
    for item in experience:
        if not (item.start_date and item.end_date and item.start_date.isdigit()):
            continue
        end = (
            int(item.end_date)
            if item.end_date.isdigit()
            else date.today().year
        )
        total += max(0, end - int(item.start_date))
    return total or None


class MockLLMService(LLMService):
    """Deterministic offline stand-in for a real LLM provider.

    Uses simple heuristics (skill lexicon, line-pattern regexes) so the full
    pipeline runs and is testable without any API keys.
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
        for line in lines:
            if not line:
                open_entry = None
                continue
            if line.startswith(_BULLET_PREFIX):
                if open_entry is not None:
                    open_entry.highlights.append(_strip_bullet(line))
                continue
            if _is_education_line(line):
                education.append(_parse_education(line))
                open_entry = None
                continue
            match = _EXP_LINE_RE.match(line)
            if match and "@" not in line:
                open_entry = ExperienceItem(
                    title=match.group("title").strip(),
                    company=match.group("company").strip(),
                    start_date=match.group("start"),
                    end_date=match.group("end"),
                )
                experience.append(open_entry)
                continue
            open_entry = None

        years_match = _YEARS_RE.search(resume_text)
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
        years_match = _YEARS_RE.search(job_description)

        responsibilities: list[str] = []
        education_requirements: list[str] = []
        in_responsibilities = False
        for line in job_description.splitlines():
            line = line.strip()
            if not line:
                in_responsibilities = False
                continue
            if "responsibilit" in line.lower():
                in_responsibilities = True
                continue
            if in_responsibilities:
                if line.startswith(_BULLET_PREFIX):
                    responsibilities.append(_strip_bullet(line))
                    continue
                in_responsibilities = False
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
