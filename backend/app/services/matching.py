"""Deterministic, rubric-based matching between a candidate and a job.

Pure comparison of the two structured models — no LLM involvement, no
randomness: identical inputs always produce an identical MatchResult.
"""

import re

from app.models.schemas import (
    CandidateProfile,
    JobRequirements,
    MatchResult,
    ScoreComponent,
    SkillGap,
)
from app.services.domain_normalization import DomainNormalizer
from app.services.education_normalization import EducationNormalizer
from app.services.experience_duration import ExperienceDurationService
from app.services.skill_normalization import (
    NormalizedRequirements,
    SkillGroup,
    SkillNormalizer,
)

REQUIRED_SKILLS_WEIGHT = 0.40
DOMAIN_FIT_WEIGHT = 0.20
EXPERIENCE_FIT_WEIGHT = 0.20
EDUCATION_FIT_WEIGHT = 0.10
PREFERRED_SKILLS_WEIGHT = 0.10

UNKNOWN_EXPERIENCE_SCORE = 50

_STOPWORDS = frozenset(
    """
    a an and are as at be by for from in into is it its of on or our over that
    the their this to under we will with you your across able using use work
    working works who whom strong good excellent plus etc
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _gap_suggestion(skill: str) -> str:
    return (
        f"Surface concrete projects or achievements involving {skill} if you truly have "
        "that experience; otherwise plan a way to build it."
    )


class MatchingService:
    """Computes MatchResult from CandidateProfile and JobRequirements.

    Rubric: required skills 40%, domain/project fit 20%, experience 20%,
    education 10%, preferred skills 10%. Each component exposes its own
    score and an explanation via MatchResult.score_breakdown.

    Skill normalization happens internally before scoring, so callers can pass
    the raw extracted models unchanged.
    """

    def __init__(
        self,
        normalizer: SkillNormalizer | None = None,
        domain_normalizer: DomainNormalizer | None = None,
        education_normalizer: EducationNormalizer | None = None,
    ) -> None:
        self._normalizer = normalizer or SkillNormalizer()
        self._domain_normalizer = domain_normalizer or DomainNormalizer()
        self._education_normalizer = education_normalizer or EducationNormalizer()

    def calculate_match(
        self, profile: CandidateProfile, requirements: JobRequirements
    ) -> MatchResult:
        # Compute total experience deterministically when the LLM did not provide it.
        if profile.total_years_experience is None and profile.experience:
            computed_years = ExperienceDurationService().total_years(profile)
            if computed_years is not None:
                profile = profile.model_copy(
                    update={"total_years_experience": computed_years}
                )

        norm_profile = self._normalizer.normalize_candidate(profile)
        norm_requirements = self._normalizer.normalize_requirements(requirements)

        components = [
            self._required_skills_component(norm_profile, norm_requirements),
            self._domain_fit_component(norm_profile, requirements),
            self._experience_fit_component(norm_profile, requirements),
            self._education_fit_component(norm_profile, requirements),
            self._preferred_skills_component(norm_profile, norm_requirements),
        ]
        overall = round(sum(component.weight * component.score for component in components))

        matched, missing, preferred_matched = self._partition_skills(
            norm_profile, norm_requirements
        )

        behavioral = sorted({group.raw for group in norm_requirements.behavioral_groups})
        compliance = sorted({group.raw for group in norm_requirements.compliance_groups})

        return MatchResult(
            overall_score=overall,
            summary=self._factual_summary(
                norm_profile, norm_requirements, overall
            ),
            matched_skills=matched,
            missing_skills=[SkillGap(skill=skill, suggestion=_gap_suggestion(skill)) for skill in missing],
            strengths=self._seed_strengths(matched, preferred_matched, profile),
            recommendations=self._seed_recommendations(missing, requirements),
            score_breakdown=components,
            behavioral_competencies=behavioral,
            compliance_requirements=compliance,
        )

    def _partition_skills(
        self, profile: CandidateProfile, requirements: NormalizedRequirements
    ) -> tuple[list[str], list[str], list[str]]:
        profile_skills = set(profile.skills)
        matched: list[str] = []
        missing: list[str] = []

        for group in requirements.required_groups:
            present = [skill for skill in group.skills if skill in profile_skills]
            if self._group_satisfied(group, profile_skills):
                matched.extend(present)
            else:
                missing.extend(self._missing_from_group(group, profile_skills))

        preferred_matched: list[str] = []
        for group in requirements.preferred_groups:
            present = [skill for skill in group.skills if skill in profile_skills]
            if self._group_satisfied(group, profile_skills):
                preferred_matched.extend(present)

        return sorted(set(matched)), sorted(set(missing)), sorted(set(preferred_matched))

    @staticmethod
    def _group_satisfied(group: SkillGroup, profile_skills: set[str]) -> bool:
        return SkillNormalizer.group_satisfied(group, profile_skills)

    @staticmethod
    def _missing_from_group(group: SkillGroup, profile_skills: set[str]) -> list[str]:
        """Human-readable missing items for an unsatisfied group."""
        if not group.is_mapped:
            # Surface the concrete unmapped requirement so it stays visible.
            return [group.raw] if group.raw else list(group.unmapped_terms)
        if group.kind == "any":
            # For OR requirements, report the whole phrase so the candidate knows
            # any one of the listed options would satisfy it.
            return [group.raw] if group.raw else sorted(
                skill for skill in group.skills if skill not in profile_skills
            )
        # AND group: report missing mapped skills and any unrecognized atomic
        # concepts that the candidate cannot satisfy.
        missing = sorted(skill for skill in group.skills if skill not in profile_skills)
        missing.extend(group.unmapped_terms)
        return missing

    def _required_skills_component(
        self, profile: CandidateProfile, requirements: NormalizedRequirements
    ) -> ScoreComponent:
        total = len(requirements.required_groups)
        if total == 0:
            return ScoreComponent(
                name="required_skills",
                weight=REQUIRED_SKILLS_WEIGHT,
                score=100,
                detail="No required skills listed in the job description.",
            )
        satisfied = sum(
            1 for group in requirements.required_groups
            if self._group_satisfied(group, set(profile.skills))
        )
        score = round(100 * satisfied / total)
        matched = sorted(
            {
                skill
                for group in requirements.required_groups
                for skill in group.skills
                if skill in set(profile.skills) and self._group_satisfied(group, set(profile.skills))
            }
        )
        names = ", ".join(matched[:6]) if matched else "none"
        extra = ""
        non_technical = requirements.behavioral_groups + requirements.compliance_groups
        if non_technical:
            extra = (
                "; "
                + str(len(non_technical))
                + " non-technical requirement(s) excluded from skill scoring"
            )
        return ScoreComponent(
            name="required_skills",
            weight=REQUIRED_SKILLS_WEIGHT,
            score=score,
            detail=f"Matched {satisfied} of {total} required skill groups ({names}).{extra}",
        )

    def _preferred_skills_component(
        self, profile: CandidateProfile, requirements: NormalizedRequirements
    ) -> ScoreComponent:
        total = len(requirements.preferred_groups)
        if total == 0:
            return ScoreComponent(
                name="preferred_skills",
                weight=PREFERRED_SKILLS_WEIGHT,
                score=100,
                detail="No preferred skills listed in the job description.",
            )
        satisfied = sum(
            1 for group in requirements.preferred_groups
            if self._group_satisfied(group, set(profile.skills))
        )
        score = round(100 * satisfied / total)
        matched = sorted(
            {
                skill
                for group in requirements.preferred_groups
                for skill in group.skills
                if skill in set(profile.skills) and self._group_satisfied(group, set(profile.skills))
            }
        )
        names = ", ".join(matched[:6]) if matched else "none"
        return ScoreComponent(
            name="preferred_skills",
            weight=PREFERRED_SKILLS_WEIGHT,
            score=score,
            detail=f"Matched {satisfied} of {total} preferred skill groups ({names}).",
        )

    def _domain_fit_component(
        self, profile: CandidateProfile, requirements: JobRequirements
    ) -> ScoreComponent:
        if not requirements.responsibilities:
            return ScoreComponent(
                name="domain_fit",
                weight=DOMAIN_FIT_WEIGHT,
                score=100,
                detail="No responsibilities listed to compare against.",
            )
        candidate_families = self._domain_normalizer.normalize_candidate(profile)
        job_families = self._domain_normalizer.normalize_job(requirements)
        diagnostics = self._domain_normalizer.score(candidate_families, job_families)

        exact = sorted(diagnostics.exact_matches)
        adjacent = sorted({pair[0] for pair in diagnostics.adjacent_matches})
        parts = []
        if exact:
            parts.append(f"exact matches: {', '.join(exact)}")
        if adjacent:
            parts.append(f"adjacent matches: {', '.join(adjacent)}")
        detail = (
            f"Candidate families: {sorted(candidate_families)}; "
            f"job families: {sorted(job_families)}; "
            + ("; ".join(parts) if parts else "no overlap")
        )
        return ScoreComponent(
            name="domain_fit",
            weight=DOMAIN_FIT_WEIGHT,
            score=diagnostics.score,
            detail=detail,
        )

    def _experience_fit_component(
        self, profile: CandidateProfile, requirements: JobRequirements
    ) -> ScoreComponent:
        required = requirements.min_years_experience
        if required is None:
            return ScoreComponent(
                name="experience_fit",
                weight=EXPERIENCE_FIT_WEIGHT,
                score=100,
                detail="No minimum years of experience specified.",
            )
        years = profile.total_years_experience
        if years is None:
            return ScoreComponent(
                name="experience_fit",
                weight=EXPERIENCE_FIT_WEIGHT,
                score=UNKNOWN_EXPERIENCE_SCORE,
                detail=(
                    f"Role requires {required}+ years; the candidate's total experience "
                    "could not be determined from the resume."
                ),
            )
        score = min(100, round(100 * years / required))
        detail = (
            f"Role requires {required}+ years; candidate has approximately {years} "
            "(calculated from resume dates)."
        )
        return ScoreComponent(
            name="experience_fit",
            weight=EXPERIENCE_FIT_WEIGHT,
            score=score,
            detail=detail,
        )

    def _education_fit_component(
        self, profile: CandidateProfile, requirements: JobRequirements
    ) -> ScoreComponent:
        education_requirements = requirements.education_requirements
        if not education_requirements:
            return ScoreComponent(
                name="education_fit",
                weight=EDUCATION_FIT_WEIGHT,
                score=100,
                detail="No education requirements listed.",
            )
        if not profile.education:
            return ScoreComponent(
                name="education_fit",
                weight=EDUCATION_FIT_WEIGHT,
                score=0,
                detail="Job requires education credentials; none found on the resume.",
            )
        candidate_norm = self._education_normalizer.normalize_candidate(profile)
        requirement_norms = self._education_normalizer.normalize_requirements(requirements)
        diagnostics = self._education_normalizer.score(candidate_norm, requirement_norms)

        detail = (
            f"Candidate level={candidate_norm.level}, fields={sorted(candidate_norm.fields)}; "
            f"required level={diagnostics.requirements[0].min_level if diagnostics.requirements else None}, "
            f"acceptable fields={sorted(diagnostics.requirements[0].acceptable_fields) if diagnostics.requirements else []}; "
            f"level_match={diagnostics.level_match}, field_match={diagnostics.field_match}"
        )
        return ScoreComponent(
            name="education_fit",
            weight=EDUCATION_FIT_WEIGHT,
            score=diagnostics.score,
            detail=detail,
        )

    def _factual_summary(
        self,
        profile: CandidateProfile,
        requirements: NormalizedRequirements,
        overall: int,
    ) -> str:
        profile_skills = set(profile.skills)
        parts = []
        if requirements.required_groups:
            satisfied = self._satisfied_group_count(
                requirements.required_groups, profile_skills
            )
            parts.append(
                f"satisfied {satisfied} of {len(requirements.required_groups)} required skill groups"
            )
        if requirements.preferred_groups:
            satisfied = self._satisfied_group_count(
                requirements.preferred_groups, profile_skills
            )
            parts.append(
                f"{satisfied} of {len(requirements.preferred_groups)} preferred skill groups"
            )
        if requirements.min_years_experience:
            parts.append(f"the role asks for {requirements.min_years_experience}+ years of experience")
        if not parts:
            parts.append("no specific requirements could be identified")
        return f"Candidate {'; '.join(parts)}. Weighted fit score: {overall}/100."

    @staticmethod
    def _satisfied_group_count(groups: tuple[SkillGroup, ...], profile_skills: set[str]) -> int:
        return sum(1 for group in groups if MatchingService._group_satisfied(group, profile_skills))

    def _seed_strengths(
        self, matched: list[str], preferred_matched: list[str], profile: CandidateProfile
    ) -> list[str]:
        strengths = []
        if matched:
            strengths.append(f"Direct experience with: {', '.join(matched[:6])}")
        if preferred_matched:
            strengths.append(f"Also brings preferred skills: {', '.join(preferred_matched[:6])}")
        if profile.experience:
            strengths.append(f"{len(profile.experience)} listed role(s) of professional experience")
        return strengths

    def _seed_recommendations(
        self, missing: list[str], requirements: JobRequirements
    ) -> list[str]:
        recommendations = []
        if missing:
            recommendations.append(
                "Address the skill gaps above before applying — even related experience is worth surfacing."
            )
        if requirements.title:
            recommendations.append(
                f'Tailor your resume summary to the "{requirements.title}" role and mirror its key terms.'
            )
        recommendations.append(
            "Quantify achievements with metrics (%, $, latency, user counts) where possible."
        )
        return recommendations
