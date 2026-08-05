"""Tests for deterministic rule evaluation from extracted signals."""
from __future__ import annotations

import pytest

from config import LLM_TIMEOUT_MS, MAX_RETRIES, MAX_TOTAL_TOKENS, TOOL_TIMEOUT_MS
from pipeline.rule_engine import RuleEngine
from pipeline.schema import ExtractedSignal, RuleEvaluation


def _signals(**overrides: float) -> list[ExtractedSignal]:
    """Build a healthy baseline signal set with selected metric overrides."""
    values = {
        "maximum_llm_latency_ms": 300.0,
        "maximum_tool_latency": 100.0,
        "failed_llm_calls": 0.0,
        "failed_tool_calls": 0.0,
        "retry_count": 0.0,
        "empty_transcript": 0.0,
        "json_errors_detected": 0.0,
        "total_tokens": 500.0,
        "transcript_turn_count": 10.0,
        "interview_duration_seconds": 600.0,
    }
    values.update(overrides)
    return [
        ExtractedSignal(signal_name=name, value=value, unit="test", source="test")
        for name, value in values.items()
    ]


def _evaluations(**overrides: float) -> dict[str, RuleEvaluation]:
    """Evaluate baseline signals and return results keyed by rule identifier."""
    return {
        evaluation.rule_id: evaluation
        for evaluation in RuleEngine().evaluate(_signals(**overrides))
    }


def _triggered_rule_ids(evaluations: dict[str, RuleEvaluation]) -> set[str]:
    """Return IDs of triggered rules while checking required reasons."""
    assert all(item.reason for item in evaluations.values())
    return {rule_id for rule_id, item in evaluations.items() if item.triggered}


def test_healthy_interview_triggers_no_rules() -> None:
    """Healthy measurements produce no triggered deterministic rules."""
    evaluations = _evaluations()

    assert _triggered_rule_ids(evaluations) == set()
    assert evaluations["high_llm_latency"].score == 0.0


def test_llm_timeout_triggers_only_latency_rule() -> None:
    """LLM latency at the configured timeout is a likely rule match."""
    evaluations = _evaluations(maximum_llm_latency_ms=LLM_TIMEOUT_MS)

    assert _triggered_rule_ids(evaluations) == {"high_llm_latency"}
    assert evaluations["high_llm_latency"].score == pytest.approx(0.75)


def test_tool_timeout_triggers_only_tool_latency_rule() -> None:
    """Tool latency at twice the timeout is a definite rule match."""
    evaluations = _evaluations(maximum_tool_latency=TOOL_TIMEOUT_MS * 2)

    assert _triggered_rule_ids(evaluations) == {"high_tool_latency"}
    assert evaluations["high_tool_latency"].score == pytest.approx(1.0)


def test_empty_transcript_triggers_empty_and_low_activity_rules() -> None:
    """Empty data is reflected by both independently measurable rules."""
    evaluations = _evaluations(empty_transcript=1.0, transcript_turn_count=0.0)

    assert _triggered_rule_ids(evaluations) == {
        "empty_transcript", "very_low_transcript_activity"
    }
    assert evaluations["empty_transcript"].score == pytest.approx(1.0)


def test_multiple_retries_triggers_retry_rule() -> None:
    """Configured retry count is a likely deterministic rule match."""
    evaluations = _evaluations(retry_count=float(MAX_RETRIES))

    assert _triggered_rule_ids(evaluations) == {"multiple_retries"}
    assert evaluations["multiple_retries"].score == pytest.approx(0.75)


def test_json_validation_error_triggers_json_rule() -> None:
    """Observed JSON errors trigger only the validation rule."""
    evaluations = _evaluations(json_errors_detected=1.0)

    assert _triggered_rule_ids(evaluations) == {"json_validation_errors"}
    assert evaluations["json_validation_errors"].score == pytest.approx(1.0)


def test_high_token_usage_triggers_token_rule() -> None:
    """Token usage above the configured threshold is a likely match."""
    evaluations = _evaluations(total_tokens=float(MAX_TOTAL_TOKENS))

    assert _triggered_rule_ids(evaluations) == {"excessive_token_usage"}
    assert evaluations["excessive_token_usage"].score == pytest.approx(0.75)


def test_mixed_failure_signals_trigger_only_expected_rules() -> None:
    """Independent rules can trigger together from mixed measurable failures."""
    evaluations = _evaluations(
        maximum_llm_latency_ms=LLM_TIMEOUT_MS * 2,
        failed_llm_calls=1.0,
        failed_tool_calls=2.0,
        retry_count=float(MAX_RETRIES + 1),
        json_errors_detected=1.0,
    )

    assert _triggered_rule_ids(evaluations) == {
        "high_llm_latency",
        "failed_llm_calls",
        "failed_tool_calls",
        "multiple_retries",
        "json_validation_errors",
    }
    assert evaluations["high_llm_latency"].score == pytest.approx(1.0)
    assert evaluations["failed_llm_calls"].score == pytest.approx(0.75)
    assert evaluations["failed_tool_calls"].score == pytest.approx(1.0)
    assert evaluations["multiple_retries"].score == pytest.approx(1.0)
