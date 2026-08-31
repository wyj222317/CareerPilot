"""Deterministic classification of free-text job requirements by category.

Separates behavioral/competency statements and compliance/privacy requirements
from concrete technical skills so the rubric scores only technical skill gaps
while still preserving and exposing the non-technical requirements.
"""

from __future__ import annotations

import re

from app.models.schemas import JobRequirements

# Experience-related phrases (years, industry background).
_EXPERIENCE_PATTERNS = [
    r"\d+\s*\+?\s*(?:years?|yrs?)",
    r"\d+\s*年(?:以上)?[^，。；\n]{0,60}经验",
    r"experience in",
    r"background in",
    r"industry experience",
]

# Education-related phrases.
_EDUCATION_PATTERNS = [
    r"(?:bachelor|master|ph\.?d|doctorate|degree|undergraduate|graduate|本科|硕士|博士|学历|学位)",
    r"(?:major|专业)",
]

# Behavioral / soft-skill / competency phrases.
_BEHAVIORAL_PATTERNS = [
    r"strong\s+(?:execution|teamwork|problem[\s\-]?solving|communication|collaboration)",
    r"execution",
    r"teamwork",
    r"team\s+spirit",
    r"goal[\s\-]?oriented",
    r"problem[\s\-]?solving",
    r"problem\s+solving",
    r"responsibility",
    r"responsible",
    r"initiative",
    r"leadership",
    r"collaboration",
    r"interpersonal",
    r"adaptability",
    r"cross-functional",
    r"stakeholder",
    r"self[\s\-]?motivated",
    r"detail[\s\-]?oriented",
    r"ownership",
    r"执行力",
    r"团队合作",
    r"团队协作",
    r"责任心",
    r"解决问题",
    r"沟通能力",
    r"抗压能力",
]

# Compliance / regulatory / safety / privacy phrases.
_COMPLIANCE_PATTERNS = [
    r"privacy",
    r"personal information",
    r"data protection",
    r"gdpr",
    r"ccpa",
    r"hipaa",
    r"sox",
    r"compliance",
    r"compliant",
    r"regulatory",
    r"regulation",
    r"secure",
    r"security",
    r"safety",
    r"safe",
    r"confidential",
    r"legal",
    r"保密",
    r"合规",
    r"安全",
    r"隐私",
    r"个人信息",
    r"数据保护",
]

_EXPERIENCE_RE = re.compile("|".join(f"(?:{p})" for p in _EXPERIENCE_PATTERNS), re.IGNORECASE)
_EDUCATION_RE = re.compile("|".join(f"(?:{p})" for p in _EDUCATION_PATTERNS), re.IGNORECASE)
_BEHAVIORAL_RE = re.compile("|".join(f"(?:{p})" for p in _BEHAVIORAL_PATTERNS), re.IGNORECASE)
_COMPLIANCE_RE = re.compile("|".join(f"(?:{p})" for p in _COMPLIANCE_PATTERNS), re.IGNORECASE)


class RequirementClassifier:
    """Classify requirement strings into technical/behavioral/compliance buckets."""

    @staticmethod
    def classify(phrase: str) -> str:
        """Return one of: technical_skill, experience_requirement, education_requirement,
        behavioral_competency, compliance_requirement.
        """
        lowered = phrase.lower()
        if _EXPERIENCE_RE.search(lowered):
            return "experience_requirement"
        if _EDUCATION_RE.search(lowered):
            return "education_requirement"
        if _COMPLIANCE_RE.search(lowered):
            return "compliance_requirement"
        if _BEHAVIORAL_RE.search(lowered):
            return "behavioral_competency"
        return "technical_skill"

    def categorize(self, requirements: JobRequirements) -> dict[str, list[str]]:
        """Categorize all requirement strings from a JobRequirements object.

        Returns a dict with keys:
        - technical_required
        - technical_preferred
        - behavioral_competencies
        - compliance_requirements
        """
        result: dict[str, list[str]] = {
            "technical_required": [],
            "technical_preferred": [],
            "behavioral_competencies": list(requirements.behavioral_competencies),
            "compliance_requirements": list(requirements.compliance_requirements),
        }
        for phrase in requirements.required_skills:
            category = self.classify(phrase)
            if category == "behavioral_competency":
                result["behavioral_competencies"].append(phrase)
            elif category == "compliance_requirement":
                result["compliance_requirements"].append(phrase)
            else:
                result["technical_required"].append(phrase)
        for phrase in requirements.preferred_skills:
            category = self.classify(phrase)
            if category == "behavioral_competency":
                result["behavioral_competencies"].append(phrase)
            elif category == "compliance_requirement":
                result["compliance_requirements"].append(phrase)
            else:
                result["technical_preferred"].append(phrase)
        return result
