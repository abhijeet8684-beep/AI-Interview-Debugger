"""Tests for deterministic template-based engineering diagnosis reasoning."""
from __future__ import annotations

from uuid import uuid4

import pytest

from pipeline.confidence_engine import ConfidenceResult
from pipeline.llm_reasoner import LLMReasoner, PromptBuilder, TemplateReasoningProvider
from pipeline.schema import (
    EvidenceRecord,
    FailureType,
    RuleEvaluation,
    Severity,
    SimilarSession,
)


def _confidence(value: float = 0.8) -> ConfidenceResult:
    """Create a deterministic confidence result for reasoning tests."""
    return ConfidenceResult(value, {"rule_score": value}, "rule_score=0.80")


def _rule(rule_id: str) -> RuleEvaluation:
    """Create one triggered deterministic rule."""
    return RuleEvaluation(rule_id=rule_id, rule_name=rule_id, triggered=True, score=0.75, reason="Test")


def _evidence(rule_id: str, severity: Severity = Severity.HIGH) -> EvidenceRecord:
    """Create evidence that references a triggered rule."""
    return EvidenceRecord(
        evidence_id=f"evidence_{rule_id}", description="Test evidence",
        source_type="signal", source_id=rule_id, rule_id=rule_id,
        severity=severity, explanation="Observed test condition.",
    )


def test_healthy_interview_returns_schema_valid_diagnosis() -> None:
    """Healthy input produces an unknown, low-severity deterministic diagnosis."""
    diagnosis = LLMReasoner().reason([], [], [], _confidence(0.0))

    assert diagnosis.root_cause == FailureType.UNKNOWN
    assert diagnosis.evidence
    assert "Confidence summary" in diagnosis.evidence[0]


@pytest.mark.parametrize(
    ("rule_id", "expected_root"),
    [
        ("high_llm_latency", FailureType.TIMEOUT),
        ("high_tool_latency", FailureType.TOOL_FAILURE),
    ],
)
def test_timeout_categories_map_to_deterministic_roots(
    rule_id: str, expected_root: FailureType
) -> None:
    """LLM and tool timeout categories use only their deterministic mappings."""
    diagnosis = LLMReasoner().reason(
        [_rule(rule_id)], [_evidence(rule_id)], [], _confidence()
    )

    assert diagnosis.root_cause == expected_root
    assert diagnosis.rules_triggered == [rule_id]
    assert diagnosis.evidence[0] == "Observed test condition."


def test_mixed_failures_include_evidence_and_historical_references() -> None:
    """Mixed deterministic inputs retain evidence and historical session IDs."""
    similar = SimilarSession(session_id=uuid4(), similarity_score=0.9)
    diagnosis = LLMReasoner().reason(
        [_rule("failed_llm_calls"), _rule("failed_tool_calls")],
        [_evidence("failed_llm_calls"), _evidence("failed_tool_calls", Severity.CRITICAL)],
        [similar], _confidence(0.9),
    )

    assert diagnosis.root_cause == FailureType.TIMEOUT
    assert diagnosis.severity == Severity.CRITICAL
    assert diagnosis.similar_sessions == [similar.session_id]
    assert len(diagnosis.evidence) == 3


def test_low_confidence_is_preserved_in_diagnosis() -> None:
    """The reasoner relays, rather than recalculates, confidence values."""
    diagnosis = LLMReasoner().reason(
        [_rule("json_validation_errors")], [_evidence("json_validation_errors")], [], _confidence(0.2)
    )

    assert diagnosis.confidence == pytest.approx(0.2)
    assert "Confidence summary" in diagnosis.evidence[-1]


def test_template_provider_is_deterministic_and_uses_injected_components() -> None:
    """The default template provider performs no external provider interaction."""
    builder = PromptBuilder()
    provider = TemplateReasoningProvider()
    prompt = builder.build([_rule("high_llm_latency")], [_evidence("high_llm_latency")], [], _confidence())

    assert provider.generate(prompt) == provider.generate(prompt)
    assert LLMReasoner(builder, provider).reason(
        [_rule("high_llm_latency")], [_evidence("high_llm_latency")], [], _confidence()
    ) == provider.generate(prompt)
