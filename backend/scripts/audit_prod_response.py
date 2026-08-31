"""Diagnostics for the saved production API response (32/100)."""
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import CandidateProfile, JobRequirements, MatchResult
from app.services.domain_normalization import DomainNormalizer
from app.services.education_normalization import EducationNormalizer
from app.services.evidence.chunking import ResumeChunker
from app.services.skill_normalization import SkillNormalizer

JSON_PATH = Path(__file__).parent / "prod_response.json"
data = json.loads(JSON_PATH.read_text())

candidate = CandidateProfile(**data["candidate"])
requirements = JobRequirements(**data["job_requirements"])
match = MatchResult(**data["match"])

print("=" * 80)
print("Production API Response Diagnostics (overall=32)")
print("=" * 80)

# Candidate summary
print("\nCandidateProfile (sanitized)")
print(f"  skills ({len(candidate.skills)}): {candidate.skills}")
print(f"  total_years_experience: {candidate.total_years_experience}")
print(f"  experience entries: {len(candidate.experience)}")
for e in candidate.experience:
    print(f"    - {e.title} @ {e.company}  {e.start_date} - {e.end_date}")
print(f"  education entries: {len(candidate.education)}")
for e in candidate.education:
    print(f"    - {e.degree} @ {e.institution}  {e.year}")

# Job summary
print("\nJobRequirements")
print(f"  required_skills ({len(requirements.required_skills)}):")
for s in requirements.required_skills:
    print(f"    - {s!r}")
print(f"  preferred_skills ({len(requirements.preferred_skills)}):")
for s in requirements.preferred_skills:
    print(f"    - {s!r}")
print(f"  min_years_experience: {requirements.min_years_experience}")
print(f"  responsibilities ({len(requirements.responsibilities)}):")
for r in requirements.responsibilities:
    print(f"    - {r!r}")
print(f"  education_requirements ({len(requirements.education_requirements)}):")
for e in requirements.education_requirements:
    print(f"    - {e!r}")

# Skill diagnostics
normalizer = SkillNormalizer()
norm_req = normalizer.normalize_requirements(requirements)
norm_cand = normalizer.normalize_candidate(candidate)
profile_skills = set(norm_cand.skills)

print("\n" + "-" * 80)
print("Required Skill Group Diagnostics")
print("-" * 80)
for group in norm_req.required_groups:
    if not group.is_mapped:
        diag = {
            "raw": group.raw,
            "atoms": [],
            "unmapped": list(group.unmapped_terms),
            "and_or": group.kind,
            "candidate_match": [],
            "satisfied": False,
            "reason": "no recognized skill alias in requirement",
        }
    else:
        if group.kind == "all":
            satisfied = set(group.skills).issubset(profile_skills) and not group.unmapped_terms
        else:
            satisfied = any(skill in profile_skills for skill in group.skills)
        matched = [skill for skill in group.skills if skill in profile_skills]
        missing = sorted(set(group.skills) - profile_skills)
        if group.kind == "all" and group.unmapped_terms:
            missing.extend(group.unmapped_terms)
        diag = {
            "raw": group.raw,
            "atoms": list(group.skills),
            "unmapped": list(group.unmapped_terms),
            "and_or": group.kind,
            "candidate_match": matched,
            "satisfied": satisfied,
            "reason": f"satisfied by {matched}" if satisfied else f"missing {missing}",
        }
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
    if not group.is_mapped:
        diag = {
            "raw": group.raw,
            "atoms": [],
            "unmapped": list(group.unmapped_terms),
            "and_or": group.kind,
            "candidate_match": [],
            "satisfied": False,
            "reason": "no recognized skill alias in requirement",
        }
    else:
        if group.kind == "all":
            satisfied = set(group.skills).issubset(profile_skills) and not group.unmapped_terms
        else:
            satisfied = any(skill in profile_skills for skill in group.skills)
        matched = [skill for skill in group.skills if skill in profile_skills]
        missing = sorted(set(group.skills) - profile_skills)
        if group.kind == "all" and group.unmapped_terms:
            missing.extend(group.unmapped_terms)
        diag = {
            "raw": group.raw,
            "atoms": list(group.skills),
            "unmapped": list(group.unmapped_terms),
            "and_or": group.kind,
            "candidate_match": matched,
            "satisfied": satisfied,
            "reason": f"satisfied by {matched}" if satisfied else f"missing {missing}",
        }
    print(f"RAW: {diag['raw']!r}")
    print(f"  normalized atoms: {diag['atoms']}")
    print(f"  unmapped atoms: {diag['unmapped']}")
    print(f"  AND/OR: {diag['and_or']}")
    print(f"  candidate matching atoms: {diag['candidate_match']}")
    print(f"  satisfied: {diag['satisfied']}")
    print(f"  reason: {diag['reason']}")
    print()

print(f"Normalized candidate skills: {sorted(profile_skills)}")

# Domain diagnostics
domain_norm = DomainNormalizer()
cand_families = domain_norm.normalize_candidate(candidate)
job_families = domain_norm.normalize_job(requirements)
domain_diag = domain_norm.score(cand_families, job_families)
print("\n" + "-" * 80)
print("Domain Fit Diagnostics")
print("-" * 80)
print(f"candidate families: {sorted(cand_families)}")
print(f"job families: {sorted(job_families)}")
print(f"exact overlaps: {sorted(domain_diag.exact_matches)}")
print(f"adjacent overlaps: {sorted(domain_diag.adjacent_matches)}")
print(f"score: {domain_diag.score}/100")

# Education diagnostics
edu_norm = EducationNormalizer()
cand_edu = edu_norm.normalize_candidate(candidate)
reqs_edu = edu_norm.normalize_requirements(requirements)
edu_diag = edu_norm.score(cand_edu, reqs_edu)
print("\n" + "-" * 80)
print("Education Fit Diagnostics")
print("-" * 80)
print(f"candidate level={cand_edu.level!r}  fields={sorted(cand_edu.fields)}")
for req in reqs_edu:
    print(f"requirement: min_level={req.min_level!r}  acceptable_fields={sorted(req.acceptable_fields)}")
print(f"level_match={edu_diag.level_match}  field_match={edu_diag.field_match}")
print(f"score={edu_diag.score}/100")

# Chunker diagnostics
print("\n" + "-" * 80)
print("Evidence Index / Chunker Diagnostics")
print("-" * 80)
raw_text_path = Path("/Users/wuyingjun/careerpilot/backend/scripts/prod_raw_text.txt")
raw_text = raw_text_path.read_text() if raw_text_path.exists() else ""
if raw_text:
    chunks = ResumeChunker().chunk(raw_text)
    print(f"resume text characters: {len(raw_text)}")
    print(f"chunks produced: {len(chunks)}")
    for c in chunks:
        print(f"  [{c.section}] {c.text[:80]!r}...")
else:
    print("raw text not saved; cannot chunk")

# Optimization trace
print("\n" + "-" * 80)
print("Optimization Absence Trace")
print("-" * 80)
print(f"match.evidence: {match.evidence}")
print(f"optimization: {data['optimization']}")
print("Conclusion: optimization is None because evidence_report is None.")
print("evidence_report is None because build_evidence_index produced zero chunks.")
print("Zero chunks because ResumeChunker only recognizes English section headers;")
print("the resume uses Chinese headers (教育背景, 专业能力, 工作经历, 项目经历).")
