"""Adversarial tests for multilingual factuality validation (Phase 4.1)."""

from app.models.schemas import (
    EvidenceMatch,
    EvidenceReport,
    MatchResult,
    OptimizationAction,
    ResumeEvidence,
    ResumeOptimizationResult,
    ValidationIssue,
)
from app.services.factual_concepts import extract_factual_concepts
from app.services.optimization.validator import OptimizationValidator


def _cn_evidence(
    text: str,
    evidence_id: str = "exp-0-b0",
    section: str = "experience",
    entry_index: str = "0",
    title: str = "",
) -> ResumeEvidence:
    return ResumeEvidence(
        id=evidence_id,
        text=text,
        section=section,
        source="Experience",
        metadata={"entry_index": entry_index, "title": title},
    )


def _build_report(claim_text: str, support_level: str, evidence_ids: list[str]) -> EvidenceReport:
    evidence = [
        EvidenceMatch(evidence_id=eid, text="", section="experience", score=1.0)
        for eid in evidence_ids
    ]
    return EvidenceReport(
        claims=[
            {
                "claim": claim_text,
                "claim_type": "required_skill",
                "support_level": support_level,
                "evidence": evidence,
            }
        ]
    )


def _validate(evidence_list, action):
    report = _build_report("test claim", "supported", [e.id for e in evidence_list])
    result = ResumeOptimizationResult(
        actions=[action],
        rewritten_bullets=[],
        gap_warnings=[],
        validation_issues=[],
        summary="test",
    )
    match = MatchResult(
        overall_score=50,
        summary="test",
        matched_skills=[],
        missing_skills=[],
        strengths=[],
        recommendations=[],
        score_breakdown=[],
    )
    validator = OptimizationValidator(evidence_list)
    return validator.validate(result, report, match)


# ── Unit tests for extract_factual_concepts ──


def test_concept_extraction_cjk_longest_first():
    concepts = extract_factual_concepts("模型部署")
    assert "model_deployment" in concepts
    assert "deployment" not in concepts


def test_concept_extraction_stylistic_excluded():
    concepts = extract_factual_concepts("负责参与熟悉具备相关良好有效专业")
    assert len(concepts) == 0


def test_concept_extraction_english_boundary():
    concepts = extract_factual_concepts("deployed services")
    assert "deployment" in concepts
    concepts2 = extract_factual_concepts("redeployed services")
    assert "deployment" not in concepts2


def test_concept_extraction_empty_input():
    assert extract_factual_concepts("") == frozenset()
    assert extract_factual_concepts("   ") == frozenset()


def test_concept_extraction_mixed_bilingual():
    concepts = extract_factual_concepts("使用Python进行数据采集和模型训练")
    assert "data_acquisition" in concepts
    assert "model_training" in concepts


def test_concept_extraction_activity_concepts():
    concepts = extract_factual_concepts("设备维护和调试")
    assert "equipment_maintenance" in concepts
    assert "debugging" in concepts


def test_concept_extraction_stability_optimization():
    concepts = extract_factual_concepts("稳定性优化")
    assert "stability_optimization" in concepts
    assert "optimization" not in concepts


# ── Chinese adversarial cases (must FAIL validation) ──


def test_cn_adds_stability_optimization():
    """Evidence: maintenance+debugging. Optimized adds 稳定性优化."""
    ev = _cn_evidence("负责 AGV、STK、OHT 设备维护和调试")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 AGV、STK、OHT 设备维护和调试",
        optimized_text="负责 AGV、STK、OHT 设备维护、调试及稳定性优化",
        rationale="Strengthen bullet",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 0
    assert any(i.field == "unsupported_concept" for i in result.validation_issues)
    msg = next(i.message for i in result.validation_issues if i.field == "unsupported_concept")
    assert "stability_optimization" in msg


def test_cn_adds_deployment_and_production():
    """Evidence: 新设备安装. Optimized: 系统部署上线及生产环境交付."""
    ev = _cn_evidence("负责新设备安装")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责新设备安装",
        optimized_text="负责系统部署上线及生产环境交付",
        rationale="Align with JD",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 0
    assert any(i.field == "unsupported_concept" for i in result.validation_issues)
    msg = next(i.message for i in result.validation_issues if i.field == "unsupported_concept")
    assert "deployment" in msg or "production_environment" in msg or "delivery" in msg


def test_cn_adds_model_deployment():
    """Evidence: Python数据分析. Optimized adds 模型部署."""
    ev = _cn_evidence("使用 Python 进行数据分析")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="使用 Python 进行数据分析",
        optimized_text="使用 Python 进行数据分析和模型部署",
        rationale="Strengthen",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 0
    assert any(i.field == "unsupported_concept" for i in result.validation_issues)
    msg = next(i.message for i in result.validation_issues if i.field == "unsupported_concept")
    assert "model_deployment" in msg


def test_cn_cross_role_provenance():
    """Job A has PLC. Job B has Python. Merging into one bullet fails."""
    ev_a = _cn_evidence("负责 PLC 设备维护和故障排查", evidence_id="exp-0-b0", entry_index="0")
    ev_b = _cn_evidence("使用 Python 进行数据分析", evidence_id="exp-1-b0", entry_index="1")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 PLC 设备维护和故障排查",
        optimized_text="负责 PLC 设备维护、故障排查及 Python 数据分析",
        rationale="Combine strengths",
        target_requirements=["test claim"],
        evidence_ids=[ev_a.id, ev_b.id],
    )
    result = _validate([ev_a, ev_b], action)
    assert len(result.actions) == 0
    assert any(i.field == "cross_scope_evidence" for i in result.validation_issues)


def test_cn_adds_network_configuration():
    """Evidence: 设备维护调试. Optimized adds 网络配置."""
    ev = _cn_evidence("负责设备维护和调试")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责设备维护和调试",
        optimized_text="负责设备维护、调试和网络配置",
        rationale="Add JD keyword",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 0
    assert any(i.field == "unsupported_concept" for i in result.validation_issues)
    msg = next(i.message for i in result.validation_issues if i.field == "unsupported_concept")
    assert "network_configuration" in msg


def test_cn_adds_privacy_compliance():
    """Evidence: maintenance. Optimized adds 隐私合规."""
    ev = _cn_evidence("负责 AGV 设备维护")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 AGV 设备维护",
        optimized_text="负责 AGV 设备维护及隐私合规管理",
        rationale="Address compliance requirement",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 0
    assert any(i.field == "unsupported_concept" for i in result.validation_issues)
    msg = next(i.message for i in result.validation_issues if i.field == "unsupported_concept")
    assert "privacy_compliance" in msg


def test_cn_adds_sensor_to_maintenance():
    """Evidence: AGV设备维护. Optimized adds 传感器."""
    ev = _cn_evidence("负责 AGV 设备维护")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 AGV 设备维护",
        optimized_text="负责 AGV 设备和传感器维护",
        rationale="Match JD sensor keyword",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 0
    assert any(i.field == "unsupported_concept" for i in result.validation_issues)
    msg = next(i.message for i in result.validation_issues if i.field == "unsupported_concept")
    assert "sensor" in msg


def test_cn_uses_non_cited_evidence_concept():
    """Concept exists in resume but NOT in cited evidence — must fail."""
    ev_cited = _cn_evidence("负责设备维护", evidence_id="exp-0-b0")
    ev_other = _cn_evidence("负责数据采集和模型训练", evidence_id="exp-1-b0")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责设备维护",
        optimized_text="负责设备维护和模型训练",
        rationale="Strengthen",
        target_requirements=["test claim"],
        evidence_ids=[ev_cited.id],
    )
    result = _validate([ev_cited, ev_other], action)
    assert len(result.actions) == 0
    assert any(i.field == "unsupported_concept" for i in result.validation_issues)
    msg = next(i.message for i in result.validation_issues if i.field == "unsupported_concept")
    assert "model_training" in msg


# ── Passing cases (must NOT trigger issues) ──


def test_cn_faithful_rewrite_passes():
    """Restyle 负责→主导, same factual concepts → passes."""
    ev = _cn_evidence("负责 AGV、STK、OHT 设备维护和调试")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 AGV、STK、OHT 设备维护和调试",
        optimized_text="主导 AGV、STK、OHT 设备维护与调试工作",
        rationale="Stronger action verb",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 1
    assert len(result.validation_issues) == 0


def test_cn_subset_rewrite_passes():
    """Dropping a concept from evidence is always safe."""
    ev = _cn_evidence("负责 AGV、STK、OHT 设备维护和调试")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 AGV、STK、OHT 设备维护和调试",
        optimized_text="负责 AGV 设备维护",
        rationale="Focus on key equipment",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 1
    assert len(result.validation_issues) == 0


def test_cn_single_scope_multi_bullet_passes():
    """Same job, multiple bullets — no cross-scope issue."""
    ev1 = _cn_evidence("负责 AGV 设备维护", evidence_id="exp-0-b0", entry_index="0")
    ev2 = _cn_evidence("调试 STK 设备", evidence_id="exp-0-b1", entry_index="0")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 AGV 设备维护",
        optimized_text="负责 AGV 和 STK 设备维护与调试",
        rationale="Combine related bullets",
        target_requirements=["test claim"],
        evidence_ids=[ev1.id, ev2.id],
    )
    result = _validate([ev1, ev2], action)
    assert len(result.actions) == 1
    assert not any(i.field == "cross_scope_evidence" for i in result.validation_issues)


def test_cn_identical_text_passes():
    """Optimized text identical to evidence — no new concepts."""
    ev = _cn_evidence("负责 AGV、STK、OHT 设备维护和调试")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="负责 AGV、STK、OHT 设备维护和调试",
        optimized_text="负责 AGV、STK、OHT 设备维护和调试",
        rationale="No change needed",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 1
    assert len(result.validation_issues) == 0


def test_english_adversarial_still_works():
    """Existing English check F still catches skill fabrication."""
    ev = ResumeEvidence(
        id="exp-0",
        text="Built APIs with FastAPI and Docker",
        section="experience",
        source="Experience",
    )
    action = OptimizationAction(
        action_type="rewrite",
        target_section="experience",
        original_text="Built APIs with FastAPI and Docker",
        optimized_text="Built APIs with FastAPI, Docker and Kubernetes on AWS",
        rationale="Match JD",
        target_requirements=["test claim"],
        evidence_ids=[ev.id],
    )
    result = _validate([ev], action)
    assert len(result.actions) == 0
    assert any("kubernetes" in i.message.lower() for i in result.validation_issues)


def test_summary_section_exempt_from_cross_scope():
    """Summary section can aggregate across roles — no cross-scope issue."""
    ev_a = _cn_evidence("负责 PLC 设备维护", evidence_id="exp-0-b0", entry_index="0")
    ev_b = _cn_evidence("使用 Python 进行开发", evidence_id="exp-1-b0", entry_index="1")
    action = OptimizationAction(
        action_type="rewrite",
        target_section="summary",
        original_text="自动化工程师",
        optimized_text="自动化工程师，负责 PLC 设备维护和 Python 开发",
        rationale="Enrich summary",
        target_requirements=["test claim"],
        evidence_ids=[ev_a.id, ev_b.id],
    )
    result = _validate([ev_a, ev_b], action)
    assert len(result.actions) == 1
    assert not any(i.field == "cross_scope_evidence" for i in result.validation_issues)
