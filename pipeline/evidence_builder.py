"""Build structured evidence records from triggered deterministic rules."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from config import (
    LLM_TIMEOUT_MS,
    LONG_INTERVIEW_SECONDS,
    MAX_RETRIES,
    MAX_TOTAL_TOKENS,
    MIN_TRANSCRIPT_TURNS,
    TOOL_TIMEOUT_MS,
)
from pipeline.schema import EvidenceRecord, ExtractedSignal, RuleEvaluation, Severity


class EvidenceBuilder:
    """Create traceable, deterministic evidence for triggered rule evaluations."""

    def build(
        self,
        signals: List[ExtractedSignal],
        rules: List[RuleEvaluation],
    ) -> List[EvidenceRecord]:
        """Build one evidence record for every triggered rule.

        Args:
            signals: Observable metrics produced by signal extraction.
            rules: Deterministic results produced by the rule engine.

        Returns:
            Structured evidence records for triggered rules only.
        """
        signal_values = {signal.signal_name: signal.value for signal in signals}
        return [
            self._build_evidence(rule, signal_values)
            for rule in rules
            if rule.triggered
        ]

    def _build_evidence(
        self, rule: RuleEvaluation, signal_values: Dict[str, float]
    ) -> EvidenceRecord:
        """Build one source-linked evidence record for a triggered rule."""
        metric, expected, severity, explanation = self._evidence_context(rule.rule_id)
        observed = signal_values.get(metric, 0.0)
        return EvidenceRecord(
            evidence_id=f"evidence_{rule.rule_id}",
            description=rule.rule_name,
            source_type="extracted_signal",
            source_id=metric,
            metric=metric,
            observed_value=observed,
            expected_value=expected,
            severity=severity,
            confidence=rule.score,
            explanation=explanation,
            signal_name=metric,
            signal_value=observed,
            rule_id=rule.rule_id,
        )

    def _evidence_context(
        self, rule_id: str
    ) -> Tuple[str, float, Severity, str]:
        """Return deterministic metric, threshold, severity, and explanation."""
        contexts = {
            "high_llm_latency": (
                "maximum_llm_latency_ms", LLM_TIMEOUT_MS, Severity.HIGH,
                "Maximum LLM latency exceeded the configured threshold.",
            ),
            "high_tool_latency": (
                "maximum_tool_latency", TOOL_TIMEOUT_MS, Severity.HIGH,
                "Maximum tool latency exceeded the configured threshold.",
            ),
            "failed_llm_calls": (
                "failed_llm_calls", 1.0, Severity.HIGH,
                "One or more LLM calls reported an explicit failure status.",
            ),
            "failed_tool_calls": (
                "failed_tool_calls", 1.0, Severity.HIGH,
                "One or more tool calls reported an explicit failure status.",
            ),
            "multiple_retries": (
                "retry_count", float(MAX_RETRIES), Severity.MEDIUM,
                "Retry activity reached the configured retry threshold.",
            ),
            "empty_transcript": (
                "empty_transcript", 1.0, Severity.HIGH,
                "The transcript is empty or contains an empty turn.",
            ),
            "json_validation_errors": (
                "json_errors_detected", 1.0, Severity.MEDIUM,
                "A JSON validation error was observed in call telemetry.",
            ),
            "excessive_token_usage": (
                "total_tokens", float(MAX_TOTAL_TOKENS), Severity.MEDIUM,
                "Total LLM token usage exceeded the configured threshold.",
            ),
            "very_low_transcript_activity": (
                "transcript_turn_count", float(MIN_TRANSCRIPT_TURNS), Severity.MEDIUM,
                "Transcript activity is below the configured minimum turn count.",
            ),
            "long_interview_duration": (
                "interview_duration_seconds", LONG_INTERVIEW_SECONDS, Severity.LOW,
                "Interview duration exceeded the configured threshold.",
            ),
        }
        return contexts[rule_id]
