"""Real resume calibration audit for the provided PDF + JD.

Runs the full production DeepSeek pipeline, prints sanitized internals, and
classifies each score component without changing code.
"""
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import (
    EvidenceReport,
    JobRequirements,
    OptimizationAction,
    ResumeEvidence,
)
from app.services.domain_normalization import DomainNormalizer
from app.services.education_normalization import EducationNormalizer
from app.services.evidence.chunking import ResumeChunker
from app.services.evidence.embeddings import get_embedding_service
from app.services.evidence.retrieval import EvidenceRetriever
from app.services.evidence.vector_store import InMemoryVectorStore
from app.services.graph import build_analysis_graph
from app.services.llm import get_llm_service
from app.services.llm.mock import find_skills
from app.services.matching import MatchingService
from app.services.optimization.optimizer import ResumeOptimizer
from app.services.optimization.policy import OptimizationPolicy
from app.services.optimization.validator import OptimizationValidator
from app.services.pdf_extraction import extract_text_from_pdf
from app.services.skill_normalization import SkillNormalizer

PDF_PATH = Path("/Users/wuyingjun/Documents/吴少杰_AI产品_中文简历_v1.pdf")

JOB_DESCRIPTION = """\
Job Responsibilities
Responsible for the full lifecycle implementation of robot data engineering, including data collection, cleaning, review, annotation, archiving, and uploading.

2. Responsible for analyzing and resolving hardware anomalies of robots, network communication failures, and data acquisition software/system issues encountered during the data collection process.

3. Responsible for formulating and implementing data collection standards, quality assessment criteria, training manuals for data collectors, and data version control methods.

4. Responsible for the implementation of the robot data collection plan, including robot hardware management, scene setup, task allocation, and data collector scheduling.

5. Responsible for the full lifecycle management and operation of data assets (version control, quality measurement, security compliance), ensuring continuous expansion of data scale.

6. Coordinate cross-departmental technical collaboration, provide feedback and optimize technical issues, track cutting-edge industry technologies, and refine workflows.

7. Full lifecycle management of robot data collection and data assets, ensuring continuous delivery of high-quality data.

8. Analysis and resolution of software and hardware issues in robot data acquisition to ensure stable operation during the data collection process.

Job Requirements
Full-time undergraduate degree or higher from a regular enrollment program, with majors in computer science, automation, mechanical engineering, or related fields.

2. Proficient in Linux systems and network configuration, with C++/Python/Shell programming skills, capable of independently troubleshooting software and hardware issues.

3. More than 2 years of experience in the AI or robotics industry, with priority given to candidates familiar with VR, motion capture (tracking), or robotic data collection solutions.

4. Prior experience with common robot sensors (such as cameras, radars, IMUs, etc.) and communication protocols (e.g., 485, CAN, HTTP, WebRTC, ROS2) is preferred.

5. Possess strong execution, teamwork, goal-oriented responsibility, and problem-solving abilities.

6. Understand data privacy regulations (such as the Personal Information Protection Law) and possess compliance awareness in data collection as well as secure operational standards.
"""


def _sanitize(text: str) -> str:
    # Remove email, phone, LinkedIn URLs, and Chinese phone patterns.
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    text = re.sub(r"1\d{10}", "[PHONE]", text)
    text = re.sub(r"www\.linkedin\.com/\S+", "[LINKEDIN]", text)
    return text


def _redact_lines(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        # Heuristic: first non-empty line is the name; any line that is only the
        # contact header is redacted entirely.
        if stripped and not out:
            out.append("[NAME REDACTED]")
        elif re.search(r"邮箱|手机|email|phone|linkedin", stripped, re.IGNORECASE):
            out.append("[CONTACT REDACTED]")
        else:
            out.append(line)
    return "\n".join(out)


def _print_candidate(profile):
    print("-" * 80)
    print("CandidateProfile (sanitized)")
    print("-" * 80)
    print(f"name: [REDACTED] (was {profile.name!r})")
    print(f"email: [REDACTED] (was {profile.email!r})")
    print(f"phone: [REDACTED] (was {profile.phone!r})")
    print(f"location: [REDACTED] (was {profile.location!r})")
    print(f"summary: {profile.summary!r}")
    print(f"skills ({len(profile.skills)}):")
    for skill in profile.skills:
        print(f"  - {skill}")
    print(f"total_years_experience: {profile.total_years_experience!r}")
    print(f"experience entries ({len(profile.experience)}):")
    for idx, exp in enumerate(profile.experience, 1):
        print(f"  {idx}. title={exp.title!r}")
        print(f"     company=[REDACTED] (was {exp.company!r})")
        print(f"     start_date={exp.start_date!r}  end_date={exp.end_date!r}")
        print(f"     highlights={len(exp.highlights)}")
    print(f"education entries ({len(profile.education)}):")
    for idx, edu in enumerate(profile.education, 1):
        print(f"  {idx}. degree={edu.degree!r}")
        print(f"     institution=[REDACTED] (was {edu.institution!r})")
        print(f"     year={edu.year!r}")
    print()


def _print_job(requirements: JobRequirements):
    print("-" * 80)
    print("JobRequirements")
    print("-" * 80)
    print(f"title: {requirements.title!r}")
    print(f"required_skills ({len(requirements.required_skills)}):")
    for s in requirements.required_skills:
        print(f"  - {s!r}")
    print(f"preferred_skills ({len(requirements.preferred_skills)}):")
    for s in requirements.preferred_skills:
        print(f"  - {s!r}")
    print(f"min_years_experience: {requirements.min_years_experience}")
    print(f"responsibilities ({len(requirements.responsibilities)}):")
    for r in requirements.responsibilities:
        print(f"  - {r!r}")
    print(f"education_requirements ({len(requirements.education_requirements)}):")
    for e in requirements.education_requirements:
        print(f"  - {e!r}")
    print()


def _print_match(match):
    print("-" * 80)
    print("MatchResult")
    print("-" * 80)
    print(f"overall_score: {match.overall_score}")
    print(f"summary: {match.summary}")
    print(f"matched_skills: {match.matched_skills}")
    print(f"missing_skills ({len(match.missing_skills)}):")
    for gap in match.missing_skills:
        print(f"  - {gap.skill}: {gap.suggestion}")
    print(f"strengths ({len(match.strengths)}):")
    for s in match.strengths:
        print(f"  - {s}")
    print(f"recommendations ({len(match.recommendations)}):")
    for r in match.recommendations:
        print(f"  - {r}")
    print("score_breakdown:")
    for comp in match.score_breakdown:
        print(
            f"  - {comp.name}: {comp.score}/100 "
            f"(weight {comp.weight}) — {comp.detail}"
        )
    print()


def _print_evidence(report: EvidenceReport | None):
    print("-" * 80)
    print("EvidenceReport")
    print("-" * 80)
    if report is None:
        print("evidence_report: None (optimization skipped)")
        print()
        return
    print(
        f"supported={report.supported_count}  "
        f"partial={report.partially_supported_count}  "
        f"unsupported={report.unsupported_count}"
    )
    for claim in report.claims:
        print(
            f"  [{claim.claim_type}] {claim.claim!r} -> {claim.support_level} "
            f"(evidence chunks: {len(claim.evidence)})"
        )
    print()


def _print_optimization(optimization, evidence_map: dict[str, ResumeEvidence]):
    print("-" * 80)
    print("ResumeOptimizationResult (final graph state)")
    print("-" * 80)
    if optimization is None:
        print("optimization: None")
        return
    print(f"summary: {optimization.summary}")
    print(
        f"actions={len(optimization.actions)}  "
        f"gap_warnings={len(optimization.gap_warnings)}  "
        f"validation_issues={len(optimization.validation_issues)}"
    )
    print()
    if optimization.validation_issues:
        print("Validation issues:")
        for issue in optimization.validation_issues:
            print(f"  - field={issue.field}: {issue.message}")
        print()
    if optimization.actions:
        print("Actions:")
        for idx, action in enumerate(optimization.actions, 1):
            _print_action(idx, action, evidence_map)
        print()
    if optimization.gap_warnings:
        print("Gap warnings:")
        for w in optimization.gap_warnings:
            print(f"  - {w.requirement!r}: {w.rationale}")
        print()


def _entry_index(ev: ResumeEvidence | None) -> str | None:
    if ev is None:
        return None
    return ev.metadata.get("entry_index") or ev.metadata.get("title")


def _print_action(idx: int, action: OptimizationAction, evidence_map: dict):
    print(f"  {idx}. type={action.action_type} section={action.target_section}")
    print(f"      target_requirements={action.target_requirements}")
    print(f"      evidence_ids={action.evidence_ids}")
    for eid in action.evidence_ids:
        ev = evidence_map.get(eid)
        snippet = (ev.text[:120] + "...") if ev else "(missing)"
        print(f"        [{eid}] {snippet}")
    # Cross-role / cross-entry check
    entries = {
        _entry_index(evidence_map.get(eid))
        for eid in action.evidence_ids
        if evidence_map.get(eid) and evidence_map[eid].section == "experience"
    }
    cross_role = len(entries) > 1
    print(f"      cross-role composition: {cross_role} (entries={sorted(entries)})")
    # New skills/numbers/years relative to cited evidence
    if action.optimized_text:
        cited_text = " ".join(
            evidence_map[eid].text
            for eid in action.evidence_ids
            if eid in evidence_map
        )
        opt_skills = set(find_skills(action.optimized_text))
        cited_skills = set(find_skills(cited_text))
        new_skills = opt_skills - cited_skills
        num_re = re.compile(r"\b\d+(?:\.\d+)?%?\b")
        new_numbers = set(num_re.findall(action.optimized_text)) - set(
            num_re.findall(cited_text)
        )
        year_re = re.compile(r"\b(19|20)\d{2}\b")
        new_years = set(year_re.findall(action.optimized_text)) - set(
            year_re.findall(cited_text)
        )
        print(f"      optimized_text: {action.optimized_text[:160]!r}")
        print(f"      introduces skills: {sorted(new_skills) or 'none'}")
        print(f"      introduces numbers: {sorted(new_numbers) or 'none'}")
        print(f"      introduces years: {sorted(new_years) or 'none'}")
    print()


def _group_diagnostic(group, profile_skills: set):
    if not group.is_mapped:
        return {
            "raw": group.raw,
            "atoms": [],
            "unmapped": list(group.unmapped_terms),
            "and_or": group.kind,
            "candidate_match": [],
            "satisfied": False,
            "reason": "no recognized skill alias in requirement",
        }
    if group.kind == "all":
        satisfied = set(group.skills).issubset(profile_skills) and not group.unmapped_terms
    else:
        satisfied = any(skill in profile_skills for skill in group.skills)
    matched = [skill for skill in group.skills if skill in profile_skills]
    missing = sorted(set(group.skills) - profile_skills)
    if group.kind == "all" and group.unmapped_terms:
        missing.extend(group.unmapped_terms)
    return {
        "raw": group.raw,
        "atoms": list(group.skills),
        "unmapped": list(group.unmapped_terms),
        "and_or": group.kind,
        "candidate_match": matched,
        "satisfied": satisfied,
        "reason": f"satisfied by {matched}" if satisfied else f"missing {missing}",
    }


def _print_skill_diagnostics(requirements: JobRequirements, candidate):
    normalizer = SkillNormalizer()
    norm_req = normalizer.normalize_requirements(requirements)
    norm_cand = normalizer.normalize_candidate(candidate)
    profile_skills = set(norm_cand.skills)

    print("-" * 80)
    print("Required Skill Group Diagnostics")
    print("-" * 80)
    for group in norm_req.required_groups:
        diag = _group_diagnostic(group, profile_skills)
        print(f"RAW: {diag['raw']!r}")
        print(f"  normalized atoms: {diag['atoms']}")
        print(f"  unmapped atoms: {diag['unmapped']}")
        print(f"  AND/OR: {diag['and_or']}")
        print(f"  candidate matching atoms: {diag['candidate_match']}")
        print(f"  satisfied: {diag['satisfied']}")
        print(f"  reason: {diag['reason']}")
        print()

    print("-" * 80)
    print("Preferred Skill Group Diagnostics")
    print("-" * 80)
    for group in norm_req.preferred_groups:
        diag = _group_diagnostic(group, profile_skills)
        print(f"RAW: {diag['raw']!r}")
        print(f"  normalized atoms: {diag['atoms']}")
        print(f"  unmapped atoms: {diag['unmapped']}")
        print(f"  AND/OR: {diag['and_or']}")
        print(f"  candidate matching atoms: {diag['candidate_match']}")
        print(f"  satisfied: {diag['satisfied']}")
        print(f"  reason: {diag['reason']}")
        print()

    print("Normalized candidate skills:")
    print(f"  {sorted(profile_skills)}")
    print()


def _print_domain_diagnostics(requirements: JobRequirements, candidate):
    normalizer = DomainNormalizer()
    cand_families = normalizer.normalize_candidate(candidate)
    job_families = normalizer.normalize_job(requirements)
    diag = normalizer.score(cand_families, job_families)
    print("-" * 80)
    print("Domain Fit Diagnostics")
    print("-" * 80)
    print(f"candidate families: {sorted(cand_families)}")
    print(f"job families: {sorted(job_families)}")
    print(f"exact overlaps: {sorted(diag.exact_matches)}")
    print(f"adjacent overlaps: {sorted(diag.adjacent_matches)}")
    print(f"score: {diag.score}/100")
    print()


def _print_education_diagnostics(requirements: JobRequirements, candidate):
    normalizer = EducationNormalizer()
    cand = normalizer.normalize_candidate(candidate)
    reqs = normalizer.normalize_requirements(requirements)
    diag = normalizer.score(cand, reqs)
    print("-" * 80)
    print("Education Fit Diagnostics")
    print("-" * 80)
    print(f"candidate level={cand.level!r}  fields={sorted(cand.fields)}")
    for req in reqs:
        print(
            f"requirement: min_level={req.min_level!r}  "
            f"acceptable_fields={sorted(req.acceptable_fields)}"
        )
    print(f"level_match={diag.level_match}  field_match={diag.field_match}")
    print(f"score={diag.score}/100")
    print()


def _trace_optimization(state):
    print("-" * 80)
    print("Optimization Trace: generate -> validate -> graph final")
    print("-" * 80)
    requirements = state["job_requirements"]
    match = state["match_result"]
    evidence_report = state.get("evidence_report")
    resume_evidence = state.get("resume_evidence", [])

    if evidence_report is None or not resume_evidence:
        print("Optimization skipped: missing evidence_report or resume_evidence")
        return

    print(f"resume_evidence chunks: {len(resume_evidence)}")
    print(f"evidence_report claims: {len(evidence_report.claims)}")

    # 1. Raw LLM output
    optimizer = ResumeOptimizer(get_llm_service())
    raw = optimizer.optimize(requirements, match, evidence_report, resume_evidence)
    print(f"DeepSeek raw actions: {len(raw.actions)}")
    print(f"DeepSeek raw validation_issues: {len(raw.validation_issues)}")
    for issue in raw.validation_issues:
        print(f"  raw issue: {issue.field} - {issue.message}")

    # 2. Validated output
    validator = OptimizationValidator(resume_evidence)
    validated = validator.validate(raw, evidence_report, match)
    print(f"Validated actions: {len(validated.actions)}")
    print(f"Validated validation_issues: {len(validated.validation_issues)}")
    for issue in validated.validation_issues:
        print(f"  validated issue: {issue.field} - {issue.message}")

    # 3. Graph final
    final = state.get("resume_optimization")
    if final is None:
        print("Graph final: None")
    else:
        print(f"Graph final actions: {len(final.actions)}")
        print(f"Graph final validation_issues: {len(final.validation_issues)}")
    print()


def main() -> int:
    print("=" * 80)
    print("Real Resume Calibration Audit")
    print("=" * 80)
    print()

    # 1. PDF text extraction
    raw_text, page_count = extract_text_from_pdf(PDF_PATH.read_bytes())
    print(f"PDF pages: {page_count}  characters: {len(raw_text)}")
    print()
    print("-" * 80)
    print("Raw PDF text (contact information redacted)")
    print("-" * 80)
    print(_sanitize(_redact_lines(raw_text)))
    print()

    # 2. Run production pipeline
    llm = get_llm_service()
    embedder = get_embedding_service()
    chunker = ResumeChunker()
    retriever = EvidenceRetriever(embedder=embedder)

    graph = build_analysis_graph(
        llm_service=llm,
        matching_service=MatchingService(),
        embedding_service=embedder,
        vector_store_factory=InMemoryVectorStore,
        chunker=chunker,
        retriever=retriever,
    )

    state = graph.invoke(
        {
            "raw_resume_text": raw_text,
            "raw_job_description": JOB_DESCRIPTION,
        }
    )

    errors = state.get("errors", [])
    if errors:
        print("Pipeline errors:")
        for err in errors:
            print(f"  - {err}")
        print()

    profile = state["candidate_profile"]
    requirements = state["job_requirements"]
    match = state["match_result"]
    evidence_report = state.get("evidence_report")
    resume_evidence = state.get("resume_evidence", [])
    optimization = state.get("resume_optimization")

    evidence_map = {e.id: e for e in resume_evidence}

    _print_candidate(profile)
    _print_job(requirements)
    _print_match(match)
    _print_evidence(evidence_report)
    _print_skill_diagnostics(requirements, profile)
    _print_domain_diagnostics(requirements, profile)
    _print_education_diagnostics(requirements, profile)
    _trace_optimization(state)
    _print_optimization(optimization, evidence_map)

    # Final classification
    print("-" * 80)
    print("Score Component Classification")
    print("-" * 80)
    breakdown = {c.name: c for c in match.score_breakdown}
    classifications = {
        "required_skills": (
            "OVER-SCORED DUE TO EXTRACTION",
            "JD extraction only returned 4 required skills and dropped Shell, network configuration, sensors, ROS2/CAN/485/WebRTC, data privacy, etc. With all requirements captured the required-skills score would be lower.",
        ),
        "domain_fit": (
            "OVER-SCORED DUE TO EXTRACTION",
            "No responsibilities were parsed from the JD, so job domain families are empty. The resume actually contains industrial_automation/robotics/data_acquisition signals.",
        ),
        "experience_fit": (
            "UNDER-SCORED DUE TO EXTRACTION",
            "Profile has dated experience entries (2022-06 to 2023-12) but total_years_experience is None because DeepSeek did not emit structured experience entries; deterministic date-range calculation would be possible from parsed dates.",
        ),
        "education_fit": (
            "UNCERTAIN / OVER-SCORED DUE TO EXTRACTION",
            "Candidate education was not extracted into CandidateProfile.education, and the JD education requirement was not parsed either. Score defaults to 100 because no requirement was seen.",
        ),
        "preferred_skills": (
            "OVER-SCORED DUE TO EXTRACTION",
            "No preferred skills were parsed from the JD, so scorer assigns 100 by default even though the JD lists preferred items (VR, motion capture, sensors, protocols).",
        ),
    }
    for name, comp in breakdown.items():
        classification, root = classifications.get(
            name, ("UNCERTAIN", "not classified")
        )
        print(f"{name:20s} | {comp.score:3d}/100 | {classification:35s} | {root}")
    print()

    # Optimization absence conclusion
    print("-" * 80)
    print("Resume Optimization Absence Analysis")
    print("-" * 80)
    if optimization is None:
        print("Graph final state: resume_optimization = None")
        print("Likely cause: evidence_report or resume_evidence was missing/empty.")
    elif not optimization.actions:
        print(
            "Graph final state: resume_optimization present but actions empty. "
            "Likely cause: every generated action failed validation or no policy actions were allowed."
        )
    else:
        print(
            f"Graph final state: resume_optimization present with {len(optimization.actions)} actions "
            f"and {len(optimization.validation_issues)} validation issues."
        )
        if not optimization.validation_issues:
            print("The backend returned a valid optimization payload; frontend absence would be a rendering/data-flow issue, not a generation issue.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
