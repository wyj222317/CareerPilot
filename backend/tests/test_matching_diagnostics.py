"""Diagnostic regression tests for Phase 3.2 real-world matching calibration.

These tests pin down the new deterministic behavior: unknown concrete
requirements are preserved, compound OR parsing works, domain scoring uses a
deterministic taxonomy, and education scoring uses level/field normalization.
"""

import pytest

from app.models.schemas import (
    CandidateProfile,
    EducationItem,
    ExperienceItem,
    JobRequirements,
)
from app.services.matching import MatchingService
from app.services.skill_normalization import SkillGroup, SkillNormalizer

NORMALIZER = SkillNormalizer()


def _real_case_candidate() -> CandidateProfile:
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


def _real_case_job() -> JobRequirements:
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


def by_name(result):
    return {component.name: component for component in result.score_breakdown}


def test_real_case_overall_score_is_58_after_requirement_separation():
    """Score after non-technical requirements are excluded from skill scoring."""
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    assert result.overall_score == 58


def test_real_case_required_skills_score_is_29():
    """2 of 7 technical required groups are satisfied (python, robotics)."""
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    breakdown = by_name(result)
    assert breakdown["required_skills"].score == 29


def test_real_case_required_groups_include_mapped_and_unmapped():
    norm = NORMALIZER.normalize_requirements(_real_case_job())
    # 3 non-technical items moved to behavioral/competency/compliance buckets.
    assert len(norm.required_groups) == 7
    mapped = [g for g in norm.required_groups if g.is_mapped]
    unmapped = [g for g in norm.required_groups if not g.is_mapped]
    # Aliases now recognize robotics, motion capture, sensor, camera, and protocols.
    assert len(mapped) == 6
    assert len(unmapped) == 1


def test_real_case_missing_gaps_surface_unmapped_requirements():
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    gap_texts = {gap.skill for gap in result.missing_skills}
    assert "Able to independently troubleshoot hardware and software issues" in gap_texts
    assert "Experience with VR / motion capture / robot data collection" in gap_texts
    assert "Familiar with robot sensors such as camera / radar / IMU" in gap_texts
    assert "Understanding of communication protocols such as 485 / CAN / HTTP / WebRTC / ROS2" in gap_texts


def test_real_case_non_technical_requirements_exposed_separately():
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    assert "Strong execution skills / teamwork / problem-solving" in result.behavioral_competencies
    assert "Understanding of data privacy regulations" in result.compliance_requirements
    assert "Awareness of compliant and safe data collection practices" in result.compliance_requirements


def test_c_python_shell_parsed_as_or_with_all_aliases():
    group = NORMALIZER.parse_claim("C / Python / Shell programming")
    assert group is not None
    assert group.kind == "any"
    assert set(group.skills) == {"c", "python", "shell"}


def test_python_only_satisfies_c_python_shell_or_group():
    profile = CandidateProfile(skills=["Python"])
    job = JobRequirements(required_skills=["C / Python / Shell programming"])
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["required_skills"].score == 100
    assert result.missing_skills == []


def test_comma_list_defaults_to_and():
    group = NORMALIZER.parse_claim("Python, Docker, Kubernetes")
    assert group is not None
    assert group.kind == "all"
    assert set(group.skills) == {"docker", "kubernetes", "python"}


def test_trailing_or_parses_comma_list_as_or():
    group = NORMALIZER.parse_claim("Python, Java, or Go")
    assert group is not None
    assert group.kind == "any"
    assert set(group.skills) == {"java", "python"}
    # "Go" is not a known alias, so it is not included in the normalized skills.


def test_and_or_phrase_parsed_as_or():
    group = NORMALIZER.parse_claim("Proficient in Python and/or Go")
    assert group is not None
    assert group.kind == "any"
    assert group.skills == ("python",)


def test_unknown_concrete_skill_preserved_and_unsatisfied():
    profile = CandidateProfile(skills=["Python"])
    job = JobRequirements(required_skills=["Experience with EtherCAT"])
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["required_skills"].score == 0
    assert any("EtherCAT" in gap.skill for gap in result.missing_skills)


def test_unknown_requirement_cannot_disappear_and_inflate_score():
    profile = CandidateProfile(skills=["Python"])
    job = JobRequirements(
        required_skills=["Python", "Experience with EtherCAT"]
    )
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["required_skills"].score == 50
    assert any("EtherCAT" in gap.skill for gap in result.missing_skills)


def test_bare_can_in_prose_does_not_match_can_protocol():
    group = NORMALIZER.parse_claim("Able to independently troubleshoot hardware issues")
    assert group is not None
    assert group.is_mapped is False
    assert "can" not in group.unmapped_terms


def test_c_alias_does_not_match_inside_words():
    group = NORMALIZER.parse_claim("Experience with C programming")
    assert group is not None
    assert group.is_mapped is True
    assert "c" in group.skills
    # "programming" should not be extracted as a skill.
    assert "programming" not in group.skills


def test_partially_mapped_and_group_unsatisfied_by_unmapped_term():
    profile = CandidateProfile(skills=["Python"])
    job = JobRequirements(required_skills=["Python and EtherCAT"])
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["required_skills"].score == 0
    gap_texts = {gap.skill for gap in result.missing_skills}
    assert "ethercat" in gap_texts


def test_partially_mapped_or_group_satisfied_by_mapped_skill():
    profile = CandidateProfile(skills=["Python"])
    job = JobRequirements(required_skills=["Python or EtherCAT"])
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["required_skills"].score == 100
    assert result.missing_skills == []


def test_linux_network_configuration_retains_both_concepts():
    group = NORMALIZER.parse_claim("Proficient in Linux systems and network configuration")
    assert group is not None
    assert group.is_mapped is True
    assert "linux" in group.skills
    assert "network configuration" in group.unmapped_terms
    assert group.kind == "all"


def test_unmapped_atoms_appear_in_diagnostics_for_partially_mapped():
    profile = CandidateProfile(skills=["Linux"])
    job = JobRequirements(required_skills=["Linux systems and network configuration"])
    result = MatchingService().calculate_match(profile, job)
    gap_texts = {gap.skill for gap in result.missing_skills}
    assert "network configuration" in gap_texts


def test_filler_words_not_surfaced_as_unmapped_atoms():
    group = NORMALIZER.parse_claim("Proficient in Linux systems and network configuration")
    assert "proficient" not in group.unmapped_terms
    assert "systems" not in group.unmapped_terms
    assert "in" not in group.unmapped_terms
    assert "and" not in group.unmapped_terms


def test_uppercase_can_and_can_bus_recognized():
    assert "can" in NORMALIZER.extract_canonical_skills("Experience with CAN bus")
    assert "can" in NORMALIZER.extract_canonical_skills("Knowledge of CAN protocol")
    assert "can" in NORMALIZER.extract_canonical_skills("Controller Area Network")
    group = NORMALIZER.parse_claim("Protocols: 485 / CAN / HTTP")
    assert group is not None
    assert "can" in group.skills


def test_lowercase_can_not_recognized_as_protocol():
    assert "can" not in NORMALIZER.extract_canonical_skills("Able to work independently")
    group = NORMALIZER.parse_claim("Able to independently troubleshoot hardware issues")
    assert group is not None
    assert "can" not in group.skills
    assert "can" not in group.unmapped_terms


def test_real_case_domain_fit_uses_taxonomy():
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    breakdown = by_name(result)
    assert breakdown["domain_fit"].score == 83
    assert "industrial_automation" in breakdown["domain_fit"].detail
    assert "robotics" in breakdown["domain_fit"].detail


def test_industrial_automation_has_adjacency_to_robotics():
    profile = CandidateProfile(
        skills=["PLC", "industrial automation"],
        experience=[ExperienceItem(title="Automation Engineer", highlights=[])],
    )
    job = JobRequirements(
        responsibilities=["Work with robotics and data collection systems."]
    )
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["domain_fit"].score > 0


def test_completely_unrelated_domains_remain_low():
    profile = CandidateProfile(
        skills=["accounting", "sales"],
        experience=[ExperienceItem(title="Sales Rep", highlights=["closed deals"])],
    )
    job = JobRequirements(
        responsibilities=["Design ROS2-based robot motion control algorithms."]
    )
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["domain_fit"].score == 0


def test_real_case_education_fit_is_100_after_normalization():
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    breakdown = by_name(result)
    assert breakdown["education_fit"].score == 100


def test_it_education_matches_computing_family():
    profile = CandidateProfile(
        education=[EducationItem(degree="M.S. Information Technology")]
    )
    job = JobRequirements(
        education_requirements=["Bachelor's degree in Computer Science or related field"]
    )
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["education_fit"].score == 100


def test_master_satisfies_bachelor_or_above():
    profile = CandidateProfile(
        education=[EducationItem(degree="M.S. Computer Science")]
    )
    job = JobRequirements(
        education_requirements=["Bachelor's degree or above in Computer Science"]
    )
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["education_fit"].score == 100


def test_unrelated_education_field_does_not_get_full_credit():
    profile = CandidateProfile(
        education=[EducationItem(degree="B.S. Mechanical Engineering")]
    )
    job = JobRequirements(
        education_requirements=["Bachelor's degree in Computer Science or related field"]
    )
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["education_fit"].score == 50


def test_real_case_preferred_skills_empty_scores_100():
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    breakdown = by_name(result)
    assert breakdown["preferred_skills"].score == 100


def test_empty_preferred_skills_always_scores_100():
    profile = CandidateProfile(skills=["Python"])
    job = JobRequirements(preferred_skills=[])
    result = MatchingService().calculate_match(profile, job)
    assert by_name(result)["preferred_skills"].score == 100


def test_real_case_experience_fit_is_50():
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    breakdown = by_name(result)
    assert breakdown["experience_fit"].score == 50


def test_soft_skill_requirement_preserved_as_unmapped():
    group = NORMALIZER.parse_claim("Strong execution skills / teamwork / problem-solving")
    assert group is not None
    assert group.is_mapped is False
    assert group.skills == ()


def test_data_privacy_requirement_preserved_as_unmapped():
    group = NORMALIZER.parse_claim("Understanding of data privacy regulations")
    assert group is not None
    assert group.is_mapped is False


def test_compliant_data_collection_requirement_preserved_as_unmapped():
    group = NORMALIZER.parse_claim("Awareness of compliant and safe data collection practices")
    assert group is not None
    assert group.is_mapped is False


def test_scoring_weights_unchanged():
    result = MatchingService().calculate_match(_real_case_candidate(), _real_case_job())
    weights = {c.name: c.weight for c in result.score_breakdown}
    assert weights["required_skills"] == 0.40
    assert weights["domain_fit"] == 0.20
    assert weights["experience_fit"] == 0.20
    assert weights["education_fit"] == 0.10
    assert weights["preferred_skills"] == 0.10


def test_scores_remain_deterministic():
    service = MatchingService()
    first = service.calculate_match(_real_case_candidate(), _real_case_job())
    second = service.calculate_match(_real_case_candidate(), _real_case_job())
    assert first == second
