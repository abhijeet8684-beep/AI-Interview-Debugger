"""Deterministically combine diagnostic evidence into confidence scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from pipeline.schema import EvidenceRecord, RuleEvaluation, Severity, SimilarSession


@dataclass(frozen=True)
class ConfidenceResult:
    """Structured deterministic confidence output for downstream consumers.

    Attributes:
        overall_confidence: Combined score bounded to the inclusive [0, 1] range.
        contributors: Named normalized components used to calculate the score.
        explanation: Deterministic summary of the component values.
    """

    overall_confidence: float
    contributors: Dict[str, float]
    explanation: str


class ConfidenceEngine:
    """Calculate a confidence score from rules, evidence, and retrieval results."""

    def calculate(
        self,
        rules: List[RuleEvaluation],
        evidence: List[EvidenceRecord],
        similar_sessions: List[SimilarSession],
    ) -> ConfidenceResult:
        """Calculate a bounded deterministic confidence result.

        Args:
            rules: Deterministic diagnostic rule evaluations.
            evidence: Structured evidence records for the current session.
            similar_sessions: Retrieved historical sessions.

        Returns:
            Confidence score, normalized contributors, and a deterministic
            component summary.
        """
        triggered_rules = [rule for rule in rules if rule.triggered]
        contributors = {
            "rule_score": self._average_rule_score(triggered_rules),
            "evidence_completeness": self._evidence_completeness(
                triggered_rules, evidence
            ),
            "evidence_severity": self._evidence_severity(evidence),
            "historical_similarity": self._historical_similarity(similar_sessions),
            "historical_failure_agreement": self._historical_agreement(
                similar_sessions
            ),
        }
        overall_confidence = self._clamp(
            0.4 * contributors["rule_score"]
            + 0.2 * contributors["evidence_completeness"]
            + 0.2 * contributors["evidence_severity"]
            + 0.1 * contributors["historical_similarity"]
            + 0.1 * contributors["historical_failure_agreement"]
        )
        return ConfidenceResult(
            overall_confidence=overall_confidence,
            contributors=contributors,
            explanation=self._explanation(contributors),
        )

    def _average_rule_score(self, rules: Sequence[RuleEvaluation]) -> float:
        """Return the mean score of triggered rules, or zero when absent."""
        scores = [rule.score or 0.0 for rule in rules]
        return self._average(scores)

    def _evidence_completeness(
        self,
        rules: Sequence[RuleEvaluation],
        evidence: Sequence[EvidenceRecord],
    ) -> float:
        """Return the fraction of triggered rules that have linked evidence."""
        if not rules:
            return 0.0
        supported_rule_ids = {record.rule_id for record in evidence if record.rule_id}
        return len({rule.rule_id for rule in rules} & supported_rule_ids) / len(rules)

    def _evidence_severity(self, evidence: Sequence[EvidenceRecord]) -> float:
        """Return the average normalized severity from available evidence."""
        values = {
            Severity.LOW: 0.25,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.75,
            Severity.CRITICAL: 1.0,
        }
        return self._average([values[record.severity] for record in evidence if record.severity])

    def _historical_similarity(self, sessions: Sequence[SimilarSession]) -> float:
        """Return the strongest available historical cosine similarity."""
        return max((session.similarity_score for session in sessions), default=0.0)

    def _historical_agreement(self, sessions: Sequence[SimilarSession]) -> float:
        """Return the share of retrieved sessions with known failure labels."""
        if not sessions:
            return 0.0
        return sum(session.failure_type is not None for session in sessions) / len(sessions)

    def _average(self, values: Sequence[float]) -> float:
        """Return a bounded average, or zero for an empty collection."""
        return self._clamp(sum(values) / len(values)) if values else 0.0

    def _clamp(self, value: float) -> float:
        """Bound a floating-point value to the inclusive [0, 1] range."""
        return max(0.0, min(1.0, value))

    def _explanation(self, contributors: Dict[str, float]) -> str:
        """Create a deterministic contributor summary without generated reasoning."""
        return (
            "Confidence contributors: "
            + ", ".join(
                f"{name}={value:.2f}" for name, value in contributors.items()
            )
            + "."
        )
