"""Evidence retrieval service.

Given JobRequirements + MatchResult + a populated VectorStore, retrieves
relevant resume evidence for each claim and classifies support levels.

Design principles:
- Retrieval ranks evidence by cosine similarity.
- Support classification for technical skills is deterministic and lexical:
  it checks whether the canonical skills required by the claim are explicitly
  evidenced across the retrieved chunks.
- Cosine similarity does NOT determine support for skill claims; it only
  selects which chunks to inspect.
"""

from app.models.schemas import (
    ClaimEvidence,
    EvidenceMatch,
    EvidenceReport,
    JobRequirements,
    MatchResult,
)
from app.services.evidence.embeddings import EmbeddingService
from app.services.evidence.vector_store import VectorStore
from app.services.skill_normalization import SkillNormalizer, SkillGroup
from app.services.text_normalization import token_set

# Thresholds for responsibility/strength claims (non-skill claims).
# These are used only when no explicit canonical skill claim can be parsed.
_RESP_SUPPORTED_OVERLAP = 0.50
_RESP_PARTIAL_OVERLAP = 0.20


class EvidenceRetriever:
    """Retrieve and classify evidence for claims from job requirements and match result."""

    def __init__(
        self,
        embedder: EmbeddingService,
        normalizer: SkillNormalizer | None = None,
        top_k: int = 4,
    ):
        self._embedder = embedder
        self._normalizer = normalizer or SkillNormalizer()
        self._top_k = top_k

    def retrieve(
        self,
        requirements: JobRequirements,
        match: MatchResult,
        store: VectorStore,
    ) -> EvidenceReport:
        """Retrieve evidence for all claims and produce an EvidenceReport."""
        claims = self._build_claims(requirements, match)
        claim_evidence_list = []

        for claim_text, claim_type in claims:
            evidence_matches = self._retrieve_for_claim(claim_text, store)
            support_level = self._classify_support(
                claim_text, claim_type, evidence_matches
            )
            claim_evidence_list.append(
                ClaimEvidence(
                    claim=claim_text,
                    claim_type=claim_type,
                    support_level=support_level,
                    evidence=evidence_matches,
                )
            )

        supported = sum(1 for c in claim_evidence_list if c.support_level == "supported")
        partial = sum(
            1 for c in claim_evidence_list if c.support_level == "partially_supported"
        )
        unsupported = sum(
            1 for c in claim_evidence_list if c.support_level == "unsupported"
        )

        return EvidenceReport(
            claims=claim_evidence_list,
            supported_count=supported,
            partially_supported_count=partial,
            unsupported_count=unsupported,
        )

    def _build_claims(
        self, requirements: JobRequirements, match: MatchResult
    ) -> list[tuple[str, str]]:
        """Build list of (claim_text, claim_type) from requirements and match."""
        claims = []
        profile_skills = set(match.matched_skills)
        norm_requirements = self._normalizer.normalize_requirements(requirements)

        # Required skill groups with at least one matched skill (includes partial)
        for group in norm_requirements.required_groups:
            if set(group.skills) & profile_skills:
                claims.append((group.raw or group.skills[0], "required_skill"))

        # Preferred skill groups with at least one matched skill
        for group in norm_requirements.preferred_groups:
            if set(group.skills) & profile_skills:
                claims.append((group.raw or group.skills[0], "preferred_skill"))

        # Strengths from match result
        for strength in match.strengths:
            claims.append((strength, "strength"))

        # Responsibilities from requirements
        for resp in requirements.responsibilities:
            claims.append((resp, "responsibility"))

        return claims

    def _retrieve_for_claim(
        self, claim_text: str, store: VectorStore
    ) -> list[EvidenceMatch]:
        """Embed claim and retrieve top_k evidence matches by cosine similarity.

        For skill claims, also performs a lexical fallback that includes any
        evidence chunk containing one of the claim's canonical skills. This
        prevents hash-based mock embeddings from missing explicit skill
        mentions (e.g. a short skill listed in the Skills section).
        """
        query_vector = self._embedder.embed([claim_text])[0]
        results = store.search(query_vector, self._top_k)
        matches = {
            ev.id: EvidenceMatch(
                evidence_id=ev.id,
                text=ev.text,
                section=ev.section,
                score=score,
            )
            for ev, score in results
        }

        group = self._normalizer.parse_claim(claim_text)
        if group is not None:
            required_skills = set(group.skills)
            for ev in store.all_evidence():
                if ev.id in matches:
                    continue
                found = set(self._normalizer.extract_canonical_skills(ev.text)) & required_skills
                if found:
                    matches[ev.id] = EvidenceMatch(
                        evidence_id=ev.id,
                        text=ev.text,
                        section=ev.section,
                        score=0.0,
                    )

        return sorted(matches.values(), key=lambda m: m.score, reverse=True)

    def _classify_support(
        self,
        claim_text: str,
        claim_type: str,
        evidence_matches: list[EvidenceMatch],
    ) -> str:
        """Classify support level for a claim based on retrieved evidence."""
        if not evidence_matches:
            return "unsupported"

        # Skill claims: deterministic lexical/canonical classification.
        if claim_type in ("required_skill", "preferred_skill"):
            return self._classify_skill_support(claim_text, evidence_matches)

        # Responsibility/strength claims: token overlap of retrieved evidence.
        return self._classify_text_support(claim_text, evidence_matches)

    def _classify_skill_support(
        self, skill_claim: str, evidence_matches: list[EvidenceMatch]
    ) -> str:
        """Classify support for a skill claim using explicit canonical skills."""
        group = self._normalizer.parse_claim(skill_claim)
        if group is None:
            # No canonical skills could be extracted from the claim;
            # fall back to text overlap (rare for skill claims).
            return self._classify_text_support(skill_claim, evidence_matches)

        # Aggregate canonical skills found across all retrieved evidence chunks.
        found_skills: set[str] = set()
        for match in evidence_matches:
            found_skills.update(self._normalizer.extract_canonical_skills(match.text))

        required_skills = set(group.skills)
        intersection = found_skills & required_skills

        if group.kind == "all":
            if required_skills.issubset(found_skills):
                return "supported"
            if intersection:
                return "partially_supported"
            return "unsupported"

        # OR requirement
        if intersection:
            return "supported"
        return "unsupported"

    def _classify_text_support(
        self, claim_text: str, evidence_matches: list[EvidenceMatch]
    ) -> str:
        """Classify support for non-skill claims using token overlap.

        Cosine similarity is used to rank evidence; overlap ratio determines
        support level. This path is NOT used for explicit technical skill
        claims.
        """
        claim_tokens = token_set(claim_text)
        if not claim_tokens:
            return "unsupported"

        evidence_text = " ".join(m.text for m in evidence_matches)
        evidence_tokens = token_set(evidence_text)
        if not evidence_tokens:
            return "unsupported"

        overlap = claim_tokens & evidence_tokens
        overlap_ratio = len(overlap) / len(claim_tokens)

        if overlap_ratio >= _RESP_SUPPORTED_OVERLAP:
            return "supported"
        if overlap_ratio >= _RESP_PARTIAL_OVERLAP:
            return "partially_supported"
        return "unsupported"
