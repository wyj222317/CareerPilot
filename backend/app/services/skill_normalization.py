"""Deterministic skill normalization between extraction and matching.

Turns raw requirement phrases and candidate skill strings into canonical,
comparable skill names and parses compound requirements into explicit AND/OR
groups. This keeps MatchingService deterministic while allowing the LLM to
emit human-readable requirement sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.models.schemas import CandidateProfile, JobRequirements
from app.services.requirement_classification import RequirementClassifier

# Single CJK unified ideograph.
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

# Alias -> canonical skill name. Canonical names stay lower-case and stable.
_SKILL_ALIASES: dict[str, str] = {
    # Core languages / runtimes
    "python": "python",
    "c": "c",
    "shell": "shell",
    "bash": "shell",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "sql": "sql",
    # Web / frontend
    "react": "react",
    "angular": "angular",
    "vue": "vue",
    # Backend / API
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "django": "django",
    "flask": "flask",
    "spring": "spring",
    "graphql": "graphql",
    "rest api": "rest api",
    "rest apis": "rest api",
    # Data stores
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mysql": "mysql",
    "mongodb": "mongodb",
    "redis": "redis",
    "pinecone": "pinecone",
    "weaviate": "weaviate",
    "chroma": "chroma",
    # LLM / agent stack
    "langchain": "langchain",
    "langgraph": "langgraph",
    "openai": "openai",
    "openai api": "openai",
    "deepseek": "deepseek",
    "deepseek api": "deepseek",
    "qwen": "qwen",
    "qwen api": "qwen",
    # Devops / cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "terraform": "terraform",
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    # Tooling / workflow
    "git": "git",
    "linux": "linux",
    "pytest": "pytest",
    "jest": "jest",
    "agile": "agile",
    "scrum": "scrum",
    "kanban": "kanban",
    "microservices": "microservices",
    "devops": "devops",
    # Data / ML
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "data science": "data science",
    "nlp": "nlp",
    "自然语言处理": "nlp",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "pandas": "pandas",
    "numpy": "numpy",
    "spark": "spark",
    "kafka": "kafka",
    # Robotics / industrial automation / embedded / data acquisition
    "ros": "ros",
    "ros2": "ros",
    "机器人操作系统": "ros",
    "imu": "imu",
    "惯性测量单元": "imu",
    "plc": "plc",
    "可编程逻辑控制器": "plc",
    "mes": "mes",
    "制造执行系统": "mes",
    "scada": "scada",
    "agv": "agv",
    "自动导引车": "agv",
    "自动导引小车": "agv",
    "motion capture": "motion capture",
    "运动捕捉": "motion capture",
    "computer vision": "computer vision",
    "计算机视觉": "computer vision",
    "data acquisition": "data acquisition",
    "数据采集": "data acquisition",
    "radar": "radar",
    "雷达": "radar",
    "毫米波雷达": "radar",
    "lidar": "lidar",
    "激光雷达": "lidar",
    "sensor": "sensor",
    "sensors": "sensor",
    "传感器": "sensor",
    "摄像头": "camera",
    "相机": "camera",
    "工业相机": "camera",
    "camera": "camera",
    "cameras": "camera",
    "robotics": "robotics",
    "机器人": "robotics",
    "工业机器人": "robotics",
    "机械臂": "robotics",
    "协作机器人": "robotics",
    "vr": "vr",
    "virtual reality": "vr",
    "虚拟现实": "vr",
    "http": "http",
    "https": "http",
    "rs485": "rs485",
    "rs-485": "rs485",
    "485": "rs485",
    "web rtc": "webrtc",
    "webrtc": "webrtc",
}

# Longest aliases first so "postgresql" wins over "postgres" and "javascript"
# wins over "js".
_SORTED_ALIASES = sorted(_SKILL_ALIASES.items(), key=lambda item: (-len(item[0]), item[0]))

# Reverse lookup: canonical skill -> set of aliases (used to strip mapped
# substrings before surfacing unmapped atomic terms).
_CANONICAL_TO_ALIASES: dict[str, set[str]] = {}
for _alias, _canonical in _SKILL_ALIASES.items():
    _CANONICAL_TO_ALIASES.setdefault(_canonical, set()).add(_alias)

# Aliases may contain spaces, slashes, dots, plus signs, etc.  The lookbehind/ahead
# ensures we do not match inside larger words (e.g. "js" inside "javascript").
_SKILL_RE = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(alias) for alias, _ in _SORTED_ALIASES)
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)

# Safe CAN protocol recognition. The inline flag (?-i:CAN) forces uppercase for
# "CAN" and its compounds so the English auxiliary word "can" is not matched.
_CAN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?-i:CAN)(?:\s+(?:bus|protocol))?\b|Controller Area Network",
    re.IGNORECASE,
)

# Words that should never surface as unknown skill atoms, even when they are
# the only concrete tokens left after stripping mapped aliases.
_UNMAPPED_FILLER_WORDS = frozenset({
    "able", "ability", "about", "above", "across", "and", "are", "as", "at",
    "be", "been", "being", "below", "but", "by", "can", "could",
    "does", "during", "each", "etc", "for", "from", "has", "have",
    "his", "how", "in", "into", "is", "its", "may", "might", "must",
    "not", "of", "off", "on", "one", "only", "onto", "or", "our",
    "out", "over", "per", "should", "shall", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "upon", "was",
    "were", "what", "when", "where", "which", "while", "who", "whom",
    "whose", "why", "will", "with", "within", "without", "would",
    "you", "your",
    # Capability / meta descriptors that are not skills
    "experience", "experienced", "familiar", "familiarity", "proficient",
    "proficiency", "strong", "understanding", "knowledge", "ability",
    "abilities", "skills", "skill", "expert", "expertise", "comfortable",
    "independently", "independent", "good", "excellent", "solid", "deep",
    "broad", "proven", "demonstrated", "practical", "hands", "level",
    # Generic nouns
    "systems", "system", "protocols", "protocol", "required", "preferred",
    "qualifications", "qualification", "responsibilities", "responsibility",
    "practices", "practice", "services", "service",
    # Generic verbs / activity descriptors
    "using", "use", "working", "work", "developing", "develop", "programming",
    "troubleshooting", "troubleshoot", "deploying", "deploy", "integrating",
    "integrate", "design", "designing",
    # Generic modifiers / awareness words
    "communication", "awareness", "compliant", "safe", "production",
    "api", "apis",
    # Chinese filler / meta descriptors
    "熟悉", "精通", "了解", "具备", "掌握", "拥有", "具有", "能够", "独立",
    "负责", "参与", "完成", "进行", "开展", "实施", "协助", "配合",
    "相关", "相关经验", "较强", "良好", "优秀", "扎实的", "丰富的",
    "等", "及", "与", "或", "如", "例如", "包括",
})

# Phrases that describe years of experience should be tracked by
# min_years_experience, not required_skills.
_EXPERIENCE_RE = re.compile(
    r"\d+\s*\+?\s*(?:years?|yrs?)\b|\d+\s*年(?:以上)?[^，。；\n]{0,60}经验",
    re.IGNORECASE,
)

# Phrases that describe education credentials should be tracked by
# education_requirements, not required_skills.
_EDUCATION_RE = re.compile(
    r"(?:\b)(degree|bachelor|master|mba|ph\.?d|b\.s\.?|m\.s\.?|b\.a\.?|m\.a\.?|undergraduate|graduate|"
    r"学历|学位|本科|学士|硕士|博士|专科|大专)(?:\b)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SkillGroup:
    """One normalized requirement: satisfy all skills, or any one of them.

    When *is_mapped* is False the group represents a concrete JD requirement
    that contains no recognized skill alias. It must remain visible and is
    never treated as satisfied.
    """

    kind: Literal["all", "any"]
    skills: tuple[str, ...]
    raw: str = ""
    is_mapped: bool = True
    unmapped_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedRequirements:
    """Job requirements after skill normalization."""

    title: str | None
    required_groups: tuple[SkillGroup, ...]
    preferred_groups: tuple[SkillGroup, ...]
    behavioral_groups: tuple[SkillGroup, ...]
    compliance_groups: tuple[SkillGroup, ...]
    min_years_experience: int | None
    responsibilities: tuple[str, ...]
    education_requirements: tuple[str, ...]


class SkillNormalizer:
    """Deterministically normalize candidate skills and job requirement phrases.

    The output is fully reproducible for the same inputs: no LLM, no randomness.
    """

    def normalize_candidate(self, profile: CandidateProfile) -> CandidateProfile:
        normalized: set[str] = set()
        for skill in profile.skills:
            normalized.update(self.extract_canonical_skills(skill))
        return profile.model_copy(update={"skills": sorted(normalized)})

    def normalize_requirements(
        self, requirements: JobRequirements
    ) -> NormalizedRequirements:
        classifier = RequirementClassifier()
        categorized = classifier.categorize(requirements)

        return NormalizedRequirements(
            title=requirements.title,
            required_groups=tuple(
                group
                for phrase in categorized["technical_required"]
                if (group := self._parse_requirement(phrase)) is not None
            ),
            preferred_groups=tuple(
                group
                for phrase in categorized["technical_preferred"]
                if (group := self._parse_requirement(phrase)) is not None
            ),
            behavioral_groups=tuple(
                group
                for phrase in categorized["behavioral_competencies"]
                if (group := self._parse_requirement(phrase)) is not None
            ),
            compliance_groups=tuple(
                group
                for phrase in categorized["compliance_requirements"]
                if (group := self._parse_requirement(phrase)) is not None
            ),
            min_years_experience=requirements.min_years_experience,
            responsibilities=tuple(requirements.responsibilities),
            education_requirements=tuple(requirements.education_requirements),
        )

    @staticmethod
    def extract_canonical_skills(text: str) -> list[str]:
        """Return sorted, unique canonical skills mentioned in *text*."""
        if not text:
            return []
        found = {
            _SKILL_ALIASES[match.group(1).lower()]
            for match in _SKILL_RE.finditer(text)
        }
        if _CAN_RE.search(text):
            found.add("can")
        return sorted(found)

    def _parse_requirement(self, phrase: str) -> SkillGroup | None:
        """Turn one requirement sentence into a normalized skill group.

        Returns ``None`` for experience/education leakage or for phrases that
        contain no recognizable skill mention.
        """
        return self.parse_claim(phrase)

    def parse_claim(self, phrase: str) -> SkillGroup | None:
        """Turn an arbitrary claim sentence into a normalized skill group.

        Recognized skills are grouped with their compound semantics. Phrases
        that contain no recognized skill alias are preserved as unmapped
        concrete requirements so they cannot silently inflate the match score.
        Returns ``None`` for experience/education leakage.
        """
        if self._is_experience_or_education(phrase):
            return None

        skills = self.extract_canonical_skills(phrase)
        kind = self._detect_compound_kind(phrase)
        if skills:
            # A requirement may mix recognized skills with unrecognized concrete
            # concepts (e.g. "Linux systems and network configuration"). Keep
            # both so the unknown part cannot silently disappear.
            unmapped_terms = tuple(self._extract_unmapped_terms(phrase, tuple(skills)))
            return SkillGroup(
                kind=kind,
                skills=tuple(skills),
                raw=phrase.strip(),
                is_mapped=True,
                unmapped_terms=unmapped_terms,
            )

        # No recognized skill alias: preserve the concrete requirement as
        # unmapped so it counts as a gap rather than disappearing.
        return SkillGroup(
            kind="all",
            skills=(),
            raw=phrase.strip(),
            is_mapped=False,
            unmapped_terms=tuple(self._extract_unmapped_terms(phrase)),
        )

    def _extract_unmapped_terms(
        self, phrase: str, mapped_skills: tuple[str, ...] = ()
    ) -> tuple[str, ...]:
        """Return concrete atomic terms from *phrase* that are not mapped skills.

        Strips generic filler words and already-recognized skill aliases so
        diagnostics only surface meaningful unknown concepts.
        """
        lowered = phrase.lower()
        # Drop experience/education-like fragments before surfacing the gap.
        lowered = _EXPERIENCE_RE.sub(" ", lowered)
        lowered = _EDUCATION_RE.sub(" ", lowered)

        # Remove mapped skill substrings so they do not reappear as unmapped.
        if mapped_skills:
            aliases_to_remove: list[str] = []
            for skill in mapped_skills:
                if skill == "can":
                    # CAN is case-sensitive; remove it with its dedicated regex
                    # so lowercase English "can" is preserved.
                    lowered = _CAN_RE.sub(" ", lowered)
                else:
                    aliases_to_remove.extend(_CANONICAL_TO_ALIASES.get(skill, set()))
            if aliases_to_remove:
                aliases_to_remove = sorted(
                    set(aliases_to_remove), key=lambda a: (-len(a), a)
                )
                alias_pattern = (
                    r"(?<![a-z0-9])("
                    + "|".join(re.escape(a) for a in aliases_to_remove)
                    + r")(?![a-z0-9])"
                )
                lowered = re.sub(alias_pattern, " ", lowered, flags=re.IGNORECASE)

        # Split on separators (including Chinese punctuation) and drop filler
        # words, keeping adjacent concrete tokens together as multi-word concepts.
        results: list[str] = []
        for chunk in re.split(r"[/,;()、，；：]+", lowered):
            # Split on non-alphanumeric non-CJK runs, then on CJK single chars.
            tokens = [t for t in re.split(r"[^a-z0-9+/\u4e00-\u9fff]+", chunk) if t]
            # Keep ASCII tokens >= 2 chars and any CJK tokens (chars or short phrases).
            clean = [
                t
                for t in tokens
                if (len(t) >= 2 and not _CJK_CHAR_RE.search(t))
                or (_CJK_CHAR_RE.search(t) and t not in _UNMAPPED_FILLER_WORDS)
            ]
            clean = [t for t in clean if t not in _UNMAPPED_FILLER_WORDS]
            if clean:
                results.append(" ".join(clean))
        return tuple(sorted(set(results)))

    @staticmethod
    def group_satisfied(group: SkillGroup, profile_skills: set[str]) -> bool:
        """Return True if *profile_skills* satisfies the skill group.

        Unmapped groups are never satisfied because the candidate has no
        recognized evidence for them. For partially-mapped AND groups, any
        unmapped atomic term makes the group unsatisfiable; for OR groups,
        unmapped terms do not contribute to satisfaction.
        """
        if not group.is_mapped:
            return False
        if group.kind == "all":
            # Unknown concrete concepts in an AND requirement cannot be ignored.
            if group.unmapped_terms:
                return False
            return set(group.skills).issubset(profile_skills)
        # OR: only recognized skills can satisfy; unmapped terms are ignored.
        return any(skill in profile_skills for skill in group.skills)

    @staticmethod
    def _is_experience_or_education(phrase: str) -> bool:
        lowered = phrase.lower()
        if _EXPERIENCE_RE.search(lowered):
            return True
        if _EDUCATION_RE.search(lowered):
            return True
        return False

    @staticmethod
    def _detect_compound_kind(phrase: str) -> Literal["all", "any"]:
        """Detect whether a phrase expresses an AND or OR requirement.

        Rules:
        - "A or B", "A and/or B", "A / B" -> OR
        - "A and B" -> AND
        - plain comma list with no conjunction -> AND
        - "A, B, or C" -> OR
        - mixed / neither -> AND (conservative default)
        """
        lowered = phrase.lower()
        if "and/or" in lowered:
            return "any"
        # Treat slash-separated alternatives as OR, e.g. "C / Python / Shell".
        normalized = re.sub(r"\s*/\s*", " or ", lowered)
        has_or = " or " in normalized
        has_and = " and " in normalized
        if has_and and not has_or:
            return "all"
        if has_or and not has_and:
            return "any"
        return "all"
