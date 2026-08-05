"""Tests for deterministic evidence construction from rule evaluations."""
from __future__ import annotations

import pytest

from config import LLM_TIMEOUT_MS, MAX_RETRIES, MAX_TOTAL_TOKENS, TOOL_TIMEOUT_MS
from pipeline.evidence_builder import EvidenceBuilder
from pipeline.schema import ExtractedSignal, RuleEvaluation, Severity


def _signal(name: str, value: float) -> ExtractedSignal:
    """Create a minimal signal for evidence-builder tests."""
    return ExtractedSignal(signal_name=name, value=value, unit="test", source="test")


def _rule(rule_id: str, score: float = 0.75, triggered: bool = True) -> RuleEvaluation:
    """Create a deterministic rule result for evidence-builder tests."""
    return RuleEvaluation(
        rule_id=rule_id,
        rule_name=rule_id.replace("_", " ").title(),
        triggered=triggered,
        score=score,
        reason="Deterministic test rule result.",
    )


def test_healthy_rules_create_no_evidence() -> None:
    """Non-triggered healthy evaluations do not create evidence cards."""
    evidence = EvidenceBuilder().build(
        [_signal("maximum_llm_latency_ms", 300.0)],
        [_rule("high_llm_latency", triggered=False)],
    )

    assert evidence == []


@pytest.mark.parametrize(
    ("rule_id", "metric", "observed", "expected", "severity"),
    [
        ("high_llm_latency", "maximum_llm_latency_ms", 31_000.0, LLM_TIMEOUT_MS, Severity.HIGH),
        ("high_tool_latency", "maximum_tool_latency", 20_000.0, TOOL_TIMEOUT_MS, Severity.HIGH),
        ("empty_transcript", "empty_transcript", 1.0, 1.0, Severity.HIGH),
        ("multiple_retries", "retry_count", 3.0, float(MAX_RETRIES), Severity.MEDIUM),
        ("json_validation_errors", "json_errors_detected", 1.0, 1.0, Severity.MEDIUM),
        ("excessive_token_usage", "total_tokens", 20_000.0, float(MAX_TOTAL_TOKENS), Severity.MEDIUM),
    ],
)
def test_triggered_rule_builds_correct_evidence(
    rule_id: str,
    metric: str,
    observed: float,
    expected: float,
    severity: Severity,
) -> None:
    """Triggered rules use their signal values and centralized thresholds."""
    evidence = EvidenceBuilder().build([_signal(metric, observed)], [_rule(rule_id)])

    assert len(evidence) == 1
    record = evidence[0]
    assert record.metric == metric
    assert record.source_id == metric
    assert record.observed_value == pytest.approx(observed)
    assert record.expected_value == pytest.approx(expected)
    assert record.severity == severity
    assert record.confidence == pytest.approx(0.75)
    assert record.explanation


def test_mixed_failure_rules_create_unique_evidence() -> None:
    """Every triggered rule creates one uniquely identified evidence record."""
    signals = [
        _signal("maximum_llm_latency_ms", 12_000.0),
        _signal("failed_tool_calls", 2.0),
        _signal("json_errors_detected", 1.0),
    ]
    rules = [
        _rule("high_llm_latency", score=1.0),
        _rule("failed_tool_calls", score=1.0),
        _rule("json_validation_errors", score=0.75),
        _rule("long_interview_duration", triggered=False),
    ]

    evidence = EvidenceBuilder().build(signals, rules)

    assert len(evidence) == 3
    assert len({record.evidence_id for record in evidence}) == 3
    assert {record.rule_id for record in evidence} == {
        "high_llm_latency", "failed_tool_calls", "json_validation_errors"
    }
    assert [record.confidence for record in evidence] == [1.0, 1.0, 0.75]
