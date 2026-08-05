"""Tests for deterministic diagnostic confidence calculation."""
from __future__ import annotations

from uuid import uuid4

import pytest

from pipeline.confidence_engine import ConfidenceEngine
from pipeline.schema import (
    EvidenceRecord,
    FailureType,
    RuleEvaluation,
    Severity,
    SimilarSession,
)
def _rule(rule_id: str, score: float, triggered: bool = True) -> RuleEvaluation:
    """Create a compact deterministic rule evaluation."""
    return RuleEvaluation(
        rule_id=rule_id, rule_name=rule_id, triggered=triggered,
        score=score, reason="Test rule.",
    )


def _evidence(rule_id: str, severity: Severity = Severity.HIGH) -> EvidenceRecord:
    """Create linked evidence with all required schema fields."""
    return EvidenceRecord(
        evidence_id=f"evidence_{rule_id}", description="Test evidence",
        source_type="signal", source_id=rule_id, rule_id=rule_id,
        severity=severity,
    )


def _similarity(score: float, known_failure: bool = True) -> SimilarSession:
    """Create one historical similarity result."""
    return SimilarSession(
        session_id=uuid4(), similarity_score=score,
        failure_type=FailureType.TIMEOUT if known_failure else None,
    )


def test_healthy_interview_has_zero_confidence() -> None:
    """No triggered evidence yields zero diagnostic confidence."""
    result = ConfidenceEngine().calculate([], [], [])

    assert result.overall_confidence == 0.0
    assert all(value == 0.0 for value in result.contributors.values())
    assert result.explanation


def test_high_confidence_uses_all_contributors() -> None:
    """Strong rules, complete evidence, and history produce high confidence."""
    engine = ConfidenceEngine()
    result = engine.calculate(
        [_rule("timeout", 1.0)], [_evidence("timeout", Severity.CRITICAL)],
        [_similarity(0.97)],
    )

    assert result.overall_confidence == pytest.approx(0.997)
    assert result.contributors["evidence_completeness"] == 1.0
    assert result.contributors["historical_similarity"] == pytest.approx(0.97)


def test_low_confidence_is_deterministic_and_bounded() -> None:
    """Weak unsupported signals remain low and repeatable."""
    engine = ConfidenceEngine()
    first = engine.calculate([_rule("weak", 0.5)], [], [])
    second = engine.calculate([_rule("weak", 0.5)], [], [])

    assert first == second
    assert first.overall_confidence == pytest.approx(0.2)
    assert 0.0 <= first.overall_confidence <= 1.0


def test_missing_evidence_reduces_completeness() -> None:
    """Triggered rules without evidence receive no completeness credit."""
    result = ConfidenceEngine().calculate([_rule("missing", 1.0)], [], [])

    assert result.contributors["evidence_completeness"] == 0.0
    assert result.overall_confidence == pytest.approx(0.4)


def test_no_historical_matches_contribute_zero_history_score() -> None:
    """Absent retrieval results contribute no similarity or agreement credit."""
    result = ConfidenceEngine().calculate(
        [_rule("timeout", 0.75)], [_evidence("timeout")], []
    )

    assert result.contributors["historical_similarity"] == 0.0
    assert result.contributors["historical_failure_agreement"] == 0.0


def test_mixed_failures_combine_multiple_objective_inputs() -> None:
    """Mixed evidence remains bounded while using every deterministic input."""
    result = ConfidenceEngine().calculate(
        [_rule("llm", 1.0), _rule("tool", 0.75)],
        [_evidence("llm", Severity.HIGH), _evidence("tool", Severity.MEDIUM)],
        [_similarity(0.9), _similarity(0.7, known_failure=False)],
    )

    assert result.contributors["rule_score"] == pytest.approx(0.875)
    assert result.contributors["historical_failure_agreement"] == pytest.approx(0.5)
    assert 0.0 <= result.overall_confidence <= 1.0
