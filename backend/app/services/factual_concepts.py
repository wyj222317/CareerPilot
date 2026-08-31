"""Deterministic bilingual factual concept extraction for optimization validation.

Identifies factual concepts (technologies, activities, domains, responsibilities)
in resume text for evidence-scoped fact checking. Stylistic words (负责, 参与,
熟悉, etc.) are excluded by design — only lexicon members surface as concepts.
"""

from __future__ import annotations

import re

from app.services.skill_normalization import _SKILL_ALIASES
from app.services.domain_normalization import DOMAIN_SIGNALS
from app.services.text_normalization import contains_cjk

_ACTIVITY_CONCEPTS: dict[str, str] = {
    # Troubleshooting / diagnostics
    "故障排查": "troubleshooting",
    "故障诊断": "troubleshooting",
    "故障处理": "troubleshooting",
    "troubleshooting": "troubleshooting",
    "troubleshoot": "troubleshooting",
    "hardware troubleshooting": "hardware_troubleshooting",
    "硬件故障排查": "hardware_troubleshooting",
    # Maintenance
    "维护": "maintenance",
    "运维": "maintenance",
    "maintenance": "maintenance",
    "maintain": "maintenance",
    "maintaining": "maintenance",
    "equipment maintenance": "equipment_maintenance",
    "设备维护": "equipment_maintenance",
    # Debugging / field work
    "调试": "debugging",
    "现场调试": "field_debugging",
    "debugging": "debugging",
    "debug": "debugging",
    "field debugging": "field_debugging",
    # Installation
    "安装": "installation",
    "installation": "installation",
    "install": "installation",
    "installing": "installation",
    # Deployment
    "部署": "deployment",
    "deployment": "deployment",
    "deploy": "deployment",
    "deployed": "deployment",
    "deploying": "deployment",
    "model deployment": "model_deployment",
    "模型部署": "model_deployment",
    # Development
    "开发": "development",
    "development": "development",
    "develop": "development",
    "developing": "development",
    # Design
    "设计": "design",
    # Optimization
    "优化": "optimization",
    "optimization": "optimization",
    "optimize": "optimization",
    "stability optimization": "stability_optimization",
    "稳定性优化": "stability_optimization",
    # Data
    "数据采集": "data_acquisition",
    "data acquisition": "data_acquisition",
    "数据分析": "data_analysis",
    "data analysis": "data_analysis",
    "模型训练": "model_training",
    "model training": "model_training",
    # Systems
    "系统集成": "system_integration",
    "system integration": "system_integration",
    "网络配置": "network_configuration",
    "network configuration": "network_configuration",
    # Compliance
    "隐私合规": "privacy_compliance",
    "数据合规": "privacy_compliance",
    "privacy compliance": "privacy_compliance",
    # Automation
    "自动化": "automation",
    "automation": "automation",
    # Production / delivery
    "生产环境": "production_environment",
    "production environment": "production_environment",
    "交付": "delivery",
    "上线": "delivery",
    "delivery": "delivery",
    # Protocols
    "通信协议": "communication_protocol",
    "communication protocol": "communication_protocol",
}


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


_cjk_aliases: dict[str, str] = {}
_ascii_aliases: dict[str, str] = {}

for _alias, _canonical in _ACTIVITY_CONCEPTS.items():
    if _has_cjk(_alias):
        _cjk_aliases[_alias] = _canonical
    else:
        _ascii_aliases[_alias] = _canonical

for _alias, _canonical in _SKILL_ALIASES.items():
    if _has_cjk(_alias) and _alias not in _cjk_aliases:
        _cjk_aliases[_alias] = f"skill:{_canonical}"

for _signal, _families in DOMAIN_SIGNALS.items():
    if _has_cjk(_signal):
        if _signal not in _cjk_aliases:
            _cjk_aliases[_signal] = f"domain:{_families[0]}"
    else:
        if _signal not in _ascii_aliases:
            _ascii_aliases[_signal] = f"domain:{_families[0]}"


def _build_cjk_pattern(aliases: dict[str, str]) -> re.Pattern | None:
    if not aliases:
        return None
    sorted_aliases = sorted(aliases.keys(), key=lambda a: (-len(a), a))
    return re.compile("|".join(re.escape(a) for a in sorted_aliases))


def _build_ascii_pattern(aliases: dict[str, str]) -> re.Pattern | None:
    if not aliases:
        return None
    sorted_aliases = sorted(aliases.keys(), key=lambda a: (-len(a), a))
    alternation = "|".join(re.escape(a) for a in sorted_aliases)
    return re.compile(
        r"(?<![a-z0-9])(" + alternation + r")(?![a-z0-9])",
        re.IGNORECASE,
    )


_CJK_PATTERN = _build_cjk_pattern(_cjk_aliases)
_ASCII_PATTERN = _build_ascii_pattern(_ascii_aliases)


def extract_factual_concepts(text: str) -> frozenset[str]:
    """Extract factual concepts from text. Returns canonical concept names."""
    if not text or not text.strip():
        return frozenset()

    concepts: set[str] = set()

    if _CJK_PATTERN:
        for match in _CJK_PATTERN.finditer(text):
            alias = match.group()
            canonical = _cjk_aliases.get(alias)
            if canonical:
                concepts.add(canonical)

    if _ASCII_PATTERN:
        for match in _ASCII_PATTERN.finditer(text):
            alias = match.group().lower()
            canonical = _ascii_aliases.get(alias)
            if canonical:
                concepts.add(canonical)

    return frozenset(concepts)
