"""Diagnostic trace of the real-world matching rubric after Phase 3.2.

This script uses a sanitized, representative fixture inspired by the observed
Chinese resume + robotics/AI data-collection JD case. It does not tune the
scorer; it only prints what the current deterministic implementation sees and
how each component score is derived.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import (
    CandidateProfile,
    EducationItem,
    ExperienceItem,
    JobRequirements,
)
from app.services.domain_normalization import DomainNormalizer
from app.services.education_normalization import EducationNormalizer
from app.services.matching import MatchingService
from app.services.skill_normalization import SkillNormalizer


def _representative_candidate() -> CandidateProfile:
    return CandidateProfile(
        name="Representative Candidate",
        summary=(
            "Industrial automation specialist with PLC, MES, AGV, STK, and OHT "
            "equipment maintenance background."
        ),
        skills=[
            "Linux",
            "Python",
            "RAG",
            "LangChain",
            "Embedding",
            "Prompt Engineering",
            "Transformer",
            "DistilBERT",
            "CNN",
            "RNN",
            "LSTM",
            "SQL",
            "Pandas",
            "React",
            "Flask",
            "Docker",
            "Git",
            "Jira",
            "MES",
            "AGV/STK/OHT",
            "PLC",
            "industrial automation",
            "robotics-related experience",
        ],
        total_years_experience=1,
        experience=[
            ExperienceItem(
                title="Automation Engineer",
                company="Example Factory",
                highlights=[
                    "Maintained PLC and MES devices",
                    "Debugged AGV, STK, and OHT machinery",
                    "Collaborated with operations staff",
                ],
            )
        ],
        education=[
            EducationItem(
                degree="M.S. Information Technology",
                institution="Example University",
                year="2024",
            )
        ],
    )


def _representative_job() -> JobRequirements:
    return JobRequirements(
        title="Robotics / AI Data Collection Engineer",
        required_skills=[
            "Proficient in Linux systems and network configuration",
            "C / Python / Shell programming",
            "Able to independently troubleshoot hardware and software issues",
            "Robotics / AI industry experience",
            "Experience with VR / motion capture / robot data collection",
            "Familiar with robot sensors such as camera / radar / IMU",
            "Understanding of communication protocols such as 485 / CAN / HTTP / WebRTC / ROS2",
            "Strong execution skills / teamwork / problem-solving",
            "Understanding of data privacy regulations",
            "Awareness of compliant and safe data collection practices",
            "2+ years of experience",
        ],
        preferred_skills=[],
        min_years_experience=2,
        responsibilities=[
            "Configure Linux systems and networks for robot data collection labs.",
            "Write C, Python, and Shell scripts to control robots and process sensor data.",
            "Independently troubleshoot hardware and software issues during data collection.",
            "Work with VR, motion capture, and robot sensors such as camera, radar, and IMU.",
            "Use communication protocols such as 485, CAN, HTTP, WebRTC, and ROS2.",
            "Follow data privacy regulations and compliant, safe data collection practices.",
            "Collaborate with the robotics and AI team to deliver high-quality datasets.",
        ],
        education_requirements=[
            "Bachelor's degree or above in Computer Science, Robotics, AI, or related field"
        ],
    )


def _required_group_diagnostic(group, profile_skills: set[str]) -> dict:
    if not group.is_mapped:
        return {
            "raw": group.raw,
            "atomic_skills": [],
            "unmapped_terms": list(group.unmapped_terms),
            "known": False,
            "kind": group.kind,
            "candidate_match": [],
            "satisfied": False,
            "reason": "no recognized skill alias in requirement",
        }

    if group.kind == "all":
        # Unmapped atomic concepts in an AND group make it impossible to satisfy.
        satisfied = set(group.skills).issubset(profile_skills) and not group.unmapped_terms
    else:
        satisfied = any(skill in profile_skills for skill in group.skills)
    matched = [skill for skill in group.skills if skill in profile_skills]
    missing = sorted(set(group.skills) - profile_skills)
    if group.kind == "all" and group.unmapped_terms:
        missing.extend(group.unmapped_terms)
    return {
        "raw": group.raw,
        "atomic_skills": list(group.skills),
        "unmapped_terms": list(group.unmapped_terms),
        "known": True,
        "kind": group.kind,
        "candidate_match": matched,
        "satisfied": satisfied,
        "reason": f"satisfied by {matched}" if satisfied else f"missing {missing}",
    }


def main() -> int:
    candidate = _representative_candidate()
    job = _representative_job()

    normalizer = SkillNormalizer()
    domain_normalizer = DomainNormalizer()
    education_normalizer = EducationNormalizer()
    matcher = MatchingService(normalizer, domain_normalizer, education_normalizer)

    norm_candidate = normalizer.normalize_candidate(candidate)
    norm_requirements = normalizer.normalize_requirements(job)

    print("=" * 70)
    print("CANDIDATE NORMALIZED PROFILE")
    print("=" * 70)
    print(f"Name: {candidate.name}")
    print(f"Total years experience: {candidate.total_years_experience}")
    print("\nOriginal skills:")
    print(f"  {candidate.skills}")
    print("\nNormalized skills (recognized aliases only):")
    print(f"  {norm_candidate.skills}")
    print("\nSummary text used for domain fit:")
    print(f"  {candidate.summary}")
    print("\nExperience titles/highlights used for domain fit:")
    for exp in candidate.experience:
        print(f"  - {exp.title}")
        for highlight in exp.highlights:
            print(f"      * {highlight}")
    print("\nEducation used for education fit:")
    for edu in candidate.education:
        print(f"  - {edu.degree} @ {edu.institution}")

    print("\n" + "=" * 70)
    print("JOB NORMALIZED REQUIREMENTS")
    print("=" * 70)
    print(f"Title: {job.title}")
    print(f"Min years experience: {job.min_years_experience}")
    print(f"\nRequired skill groups parsed ({len(norm_requirements.required_groups)}):")
    for i, group in enumerate(norm_requirements.required_groups, 1):
        status = "mapped" if group.is_mapped else "unmapped"
        print(f"  {i}. [{status}] {group.raw!r}")
        print(f"     kind={group.kind}, skills={list(group.skills)}")
    experience_like = [
        phrase
        for phrase in job.required_skills
        if normalizer._is_experience_or_education(phrase)
    ]
    print(f"\nExperience/education-like required phrases filtered out:")
    for phrase in experience_like:
        print(f"  - {phrase!r}")

    print(f"\nPreferred skill groups parsed ({len(norm_requirements.preferred_groups)}):")
    if norm_requirements.preferred_groups:
        for group in norm_requirements.preferred_groups:
            print(f"  - {group.raw!r} -> {group.kind} {list(group.skills)}")
    else:
        print("  (none)")

    print(f"\nResponsibilities used for domain fit ({len(job.responsibilities)}):")
    for resp in job.responsibilities:
        print(f"  - {resp}")

    print(f"\nEducation requirements used for education fit ({len(job.education_requirements)}):")
    for req in job.education_requirements:
        print(f"  - {req}")

    print("\n" + "=" * 70)
    print("REQUIRED SKILLS DIAGNOSTIC")
    print("=" * 70)
    profile_skills = set(norm_candidate.skills)
    required_diagnostics = [
        _required_group_diagnostic(group, profile_skills)
        for group in norm_requirements.required_groups
    ]
    for diag in required_diagnostics:
        known_label = "known" if diag["known"] else "unmapped"
        print(f"\nRaw requirement   : {diag['raw']}")
        print(f"  Known/unmapped   : {known_label}")
        print(f"  Normalized skills: {diag['atomic_skills'] or '(none)'}")
        print(f"  Unmapped atoms   : {diag['unmapped_terms'] or '(none)'}")
        print(f"  AND/OR semantics : {diag['kind']}")
        print(f"  Candidate match  : {diag['candidate_match'] or 'none'}")
        print(f"  Satisfied        : {diag['satisfied']}")
        print(f"  Reason           : {diag['reason']}")

    satisfied_count = sum(1 for d in required_diagnostics if d["satisfied"])
    total_count = len(required_diagnostics)
    required_score = round(100 * satisfied_count / total_count) if total_count else 100
    print(f"\nRequired skills score: {required_score}/100 ({satisfied_count}/{total_count} groups satisfied)")

    print("\n" + "=" * 70)
    print("DOMAIN FIT DIAGNOSTIC")
    print("=" * 70)
    candidate_families = domain_normalizer.normalize_candidate(candidate)
    job_families = domain_normalizer.normalize_job(job)
    domain_diag = domain_normalizer.score(candidate_families, job_families)
    print(f"Candidate domain families : {sorted(candidate_families)}")
    print(f"Job domain families       : {sorted(job_families)}")
    print(f"Exact overlap             : {sorted(domain_diag.exact_matches)}")
    print(
        f"Adjacent overlap          : "
        f"{sorted({pair[0] for pair in domain_diag.adjacent_matches})}"
    )
    print(f"Domain fit score          : {domain_diag.score}/100")

    print("\n" + "=" * 70)
    print("EXPERIENCE FIT DIAGNOSTIC")
    print("=" * 70)
    required_years = job.min_years_experience
    candidate_years = candidate.total_years_experience
    if required_years is None:
        exp_score = 100
        detail = "No minimum years specified."
    elif candidate_years is None:
        exp_score = 50
        detail = "Candidate years unknown; neutral fallback 50."
    else:
        exp_score = min(100, round(100 * candidate_years / required_years))
        detail = f"{candidate_years} / {required_years} -> {exp_score}"
    print(f"Required years   : {required_years}")
    print(f"Candidate years  : {candidate_years}")
    print(f"Calculation      : {detail}")
    print(f"Experience score : {exp_score}/100")

    print("\n" + "=" * 70)
    print("EDUCATION FIT DIAGNOSTIC")
    print("=" * 70)
    candidate_edu = education_normalizer.normalize_candidate(candidate)
    requirement_norms = education_normalizer.normalize_requirements(job)
    edu_diag = education_normalizer.score(candidate_edu, requirement_norms)
    print(f"Candidate degree level : {candidate_edu.level}")
    print(f"Candidate field family : {sorted(candidate_edu.fields)}")
    if requirement_norms:
        req = requirement_norms[0]
        print(f"Required minimum level : {req.min_level}")
        print(f"Acceptable field families: {sorted(req.acceptable_fields)}")
    print(f"Level match          : {edu_diag.level_match}")
    print(f"Field match          : {edu_diag.field_match}")
    print(f"Education score      : {edu_diag.score}/100")

    print("\n" + "=" * 70)
    print("PREFERRED SKILLS DIAGNOSTIC")
    print("=" * 70)
    preferred_total = len(norm_requirements.preferred_groups)
    if preferred_total == 0:
        preferred_score = 100
        print("No preferred skill groups parsed from the JD.")
        print("Current scorer assigns 100 when the list is empty.")
    else:
        preferred_satisfied = sum(
            1
            for group in norm_requirements.preferred_groups
            if (
                set(group.skills).issubset(profile_skills)
                if group.kind == "all"
                else any(skill in profile_skills for skill in group.skills)
            )
        )
        preferred_score = round(100 * preferred_satisfied / preferred_total)
        print(f"Preferred groups: {preferred_total}, satisfied: {preferred_satisfied}")
    print(f"Preferred score: {preferred_score}/100")

    print("\n" + "=" * 70)
    print("FINAL WEIGHTED SCORE")
    print("=" * 70)
    weights = {
        "required_skills": 0.40,
        "domain_fit": 0.20,
        "experience_fit": 0.20,
        "education_fit": 0.10,
        "preferred_skills": 0.10,
    }
    component_scores = {
        "required_skills": required_score,
        "domain_fit": domain_diag.score,
        "experience_fit": exp_score,
        "education_fit": edu_diag.score,
        "preferred_skills": preferred_score,
    }
    weighted_sum = 0.0
    for name, weight in weights.items():
        contribution = weight * component_scores[name]
        weighted_sum += contribution
        print(f"  {name:20s}: {component_scores[name]:3d} * {weight:.2f} = {contribution:5.1f}")
    print(f"\n  Weighted sum      : {weighted_sum:.1f}")
    print(f"  Rounded overall   : {round(weighted_sum)}/100")

    # Run the real MatchingService to confirm the trace matches the implementation.
    result = matcher.calculate_match(candidate, job)
    print("\n" + "=" * 70)
    print("MATCHINGSERVICE VERIFICATION")
    print("=" * 70)
    breakdown = {c.name: c for c in result.score_breakdown}
    for name in weights:
        comp = breakdown[name]
        print(f"  {comp.name:20s}: {comp.score:3d}/100 - {comp.detail}")
    print(f"\n  Overall score: {result.overall_score}/100")
    print(f"  Matched skills: {result.matched_skills}")
    print(f"  Missing gaps ({len(result.missing_skills)}):")
    for gap in result.missing_skills:
        print(f"    - {gap.skill}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
