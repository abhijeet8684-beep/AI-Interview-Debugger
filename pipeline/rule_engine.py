"""Evaluate deterministic diagnostic rules from extracted session signals."""
from __future__ import annotations

from typing import Dict, List, Sequence

from config import (
    LLM_TIMEOUT_MS,
    LONG_INTERVIEW_SECONDS,
    MAX_RETRIES,
    MAX_TOTAL_TOKENS,
    MIN_TRANSCRIPT_TURNS,
    TOOL_TIMEOUT_MS,
)
from pipeline.schema import ExtractedSignal, RuleEvaluation


class RuleEngine:
    """Evaluate independent deterministic rules over extracted signals."""

    def evaluate(self, signals: List[ExtractedSignal]) -> List[RuleEvaluation]:
        """Evaluate all configured rules against a collection of signals.

        Args:
            signals: Measurable signals extracted from one interview session.

        Returns:
            One evaluation for each supported deterministic rule.
        """
        values = self._signal_values(signals)
        return [
            self._evaluate_llm_latency(values),
            self._evaluate_tool_latency(values),
            self._evaluate_failed_llm_calls(values),
            self._evaluate_failed_tool_calls(values),
            self._evaluate_retries(values),
            self._evaluate_empty_transcript(values),
            self._evaluate_json_errors(values),
            self._evaluate_token_usage(values),
            self._evaluate_transcript_activity(values),
            self._evaluate_interview_duration(values),
        ]

    def _evaluate_llm_latency(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate maximum observed LLM latency."""
        return self._threshold_evaluation(
            "high_llm_latency", "High LLM latency", values.get("maximum_llm_latency_ms", 0.0),
            LLM_TIMEOUT_MS, "ms",
        )

    def _evaluate_tool_latency(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate maximum observed tool latency."""
        return self._threshold_evaluation(
            "high_tool_latency", "High Tool latency", values.get("maximum_tool_latency", 0.0),
            TOOL_TIMEOUT_MS, "ms",
        )

    def _evaluate_failed_llm_calls(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate the count of explicitly failed LLM calls."""
        return self._count_evaluation(
            "failed_llm_calls", "Failed LLM calls", values.get("failed_llm_calls", 0.0),
        )

    def _evaluate_failed_tool_calls(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate the count of explicitly failed tool calls."""
        return self._count_evaluation(
            "failed_tool_calls", "Failed Tool calls", values.get("failed_tool_calls", 0.0),
        )

    def _evaluate_retries(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate whether retry activity reached the configured maximum."""
        retries = values.get("retry_count", 0.0)
        triggered = retries >= MAX_RETRIES
        score = 1.0 if retries > MAX_RETRIES else 0.75 if triggered else 0.0
        return self._evaluation(
            "multiple_retries", "Multiple retries", triggered, score,
            f"Observed {retries:g} retries; threshold is {MAX_RETRIES}.",
        )

    def _evaluate_empty_transcript(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate the observable empty-transcript indicator."""
        empty = values.get("empty_transcript", 0.0) > 0.0
        return self._evaluation(
            "empty_transcript", "Empty transcript", empty, 1.0 if empty else 0.0,
            "Transcript is empty or contains an empty turn." if empty else "No empty transcript content observed.",
        )

    def _evaluate_json_errors(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate the observable JSON-validation-error indicator."""
        detected = values.get("json_errors_detected", 0.0) > 0.0
        return self._evaluation(
            "json_validation_errors", "JSON validation errors", detected,
            1.0 if detected else 0.0,
            "JSON validation error was observed." if detected else "No JSON validation errors observed.",
        )

    def _evaluate_token_usage(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate total LLM token usage."""
        return self._threshold_evaluation(
            "excessive_token_usage", "Excessive token usage", values.get("total_tokens", 0.0),
            MAX_TOTAL_TOKENS, "tokens",
        )

    def _evaluate_transcript_activity(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate whether transcript activity is below the configured minimum."""
        turns = values.get("transcript_turn_count", 0.0)
        triggered = turns < MIN_TRANSCRIPT_TURNS
        score = 1.0 if turns == 0 else 0.75 if triggered else 0.0
        return self._evaluation(
            "very_low_transcript_activity", "Very low transcript activity", triggered, score,
            f"Observed {turns:g} transcript turns; minimum is {MIN_TRANSCRIPT_TURNS}.",
        )

    def _evaluate_interview_duration(self, values: Dict[str, float]) -> RuleEvaluation:
        """Evaluate interview duration against the configured long-duration limit."""
        return self._threshold_evaluation(
            "long_interview_duration", "Long interview duration",
            values.get("interview_duration_seconds", 0.0), LONG_INTERVIEW_SECONDS,
            "seconds",
        )

    def _signal_values(self, signals: Sequence[ExtractedSignal]) -> Dict[str, float]:
        """Build a signal-name lookup for rule evaluation."""
        return {signal.signal_name: signal.value for signal in signals}

    def _threshold_evaluation(
        self, rule_id: str, rule_name: str, value: float, threshold: float, unit: str
    ) -> RuleEvaluation:
        """Build a threshold rule with deterministic warning and definite scores."""
        triggered = value >= threshold
        score = 1.0 if value >= threshold * 2 else 0.75 if triggered else 0.0
        return self._evaluation(
            rule_id, rule_name, triggered, score,
            f"Observed {value:g} {unit}; threshold is {threshold:g} {unit}.",
        )

    def _count_evaluation(
        self, rule_id: str, rule_name: str, count: float
    ) -> RuleEvaluation:
        """Build a count rule with likely and definite deterministic scores."""
        triggered = count > 0.0
        score = 1.0 if count >= 2.0 else 0.75 if triggered else 0.0
        return self._evaluation(
            rule_id, rule_name, triggered, score,
            f"Observed {count:g} failed calls.",
        )

    def _evaluation(
        self, rule_id: str, rule_name: str, triggered: bool, score: float, reason: str
    ) -> RuleEvaluation:
        """Create one schema-valid deterministic rule evaluation."""
        return RuleEvaluation(
            rule_id=rule_id,
            rule_name=rule_name,
            triggered=triggered,
            score=score,
            reason=reason,
        )
