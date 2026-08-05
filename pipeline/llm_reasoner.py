"""Deterministic, provider-pluggable engineering diagnosis reasoning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from pipeline.confidence_engine import ConfidenceResult
from pipeline.schema import (
    ComponentType,
    EvidenceRecord,
    FailureDiagnosis,
    FailureType,
    RuleEvaluation,
    Severity,
    SimilarSession,
)


@dataclass(frozen=True)
class ReasoningPrompt:
    """Structured deterministic reasoning context for a provider."""

    triggered_rules: List[RuleEvaluation]
    evidence: List[EvidenceRecord]
    similar_sessions: List[SimilarSession]
    confidence: ConfidenceResult
    text: str


class PromptBuilder:
    """Build reusable structured prompts from deterministic pipeline outputs."""

    def build(
        self,
        rules: Sequence[RuleEvaluation],
        evidence: Sequence[EvidenceRecord],
        similar_sessions: Sequence[SimilarSession],
        confidence: ConfidenceResult,
    ) -> ReasoningPrompt:
        """Construct a deterministic prompt without any model interaction."""
        triggered_rules = [rule for rule in rules if rule.triggered]
        prompt_text = (
            f"triggered_rules={[rule.rule_id for rule in triggered_rules]}; "
            f"evidence_count={len(evidence)}; "
            f"similar_session_count={len(similar_sessions)}; "
            f"confidence={confidence.overall_confidence:.2f}; "
            f"confidence_summary={confidence.explanation}"
        )
        return ReasoningPrompt(
            triggered_rules=triggered_rules,
            evidence=list(evidence),
            similar_sessions=list(similar_sessions),
            confidence=confidence,
            text=prompt_text,
        )


class ReasoningProvider(ABC):
    """Interface for diagnosis providers accepting structured reasoning prompts."""

    @abstractmethod
    def generate(self, prompt: ReasoningPrompt) -> FailureDiagnosis:
        """Generate a schema-valid diagnosis from deterministic prompt context."""


class TemplateReasoningProvider(ReasoningProvider):
    """Generate diagnoses through deterministic engineering templates only."""

    def generate(self, prompt: ReasoningPrompt) -> FailureDiagnosis:
        """Generate a diagnosis exclusively from supplied deterministic inputs."""
        root_cause = self._root_cause(prompt.triggered_rules)
        return FailureDiagnosis(
            root_cause=root_cause,
            affected_component=self._affected_component(prompt.triggered_rules),
            severity=self._severity(prompt.evidence),
            confidence=prompt.confidence.overall_confidence,
            evidence=self._evidence_text(prompt.evidence, prompt.confidence),
            rules_triggered=[rule.rule_id for rule in prompt.triggered_rules],
            similar_sessions=[
                session.session_id for session in prompt.similar_sessions
            ] or None,
        )

    def _root_cause(self, rules: Sequence[RuleEvaluation]) -> FailureType:
        """Map triggered deterministic rule categories to the closest taxonomy."""
        rule_ids = {rule.rule_id for rule in rules}
        if rule_ids & {"high_llm_latency", "failed_llm_calls"}:
            return FailureType.TIMEOUT
        if rule_ids & {"high_tool_latency", "failed_tool_calls"}:
            return FailureType.TOOL_FAILURE
        if "json_validation_errors" in rule_ids:
            return FailureType.PROMPT_ISSUE
        return FailureType.UNKNOWN

    def _affected_component(
        self, rules: Sequence[RuleEvaluation]
    ) -> ComponentType:
        """Map triggered rule categories to their directly observed component."""
        rule_ids = {rule.rule_id for rule in rules}
        if rule_ids & {"high_llm_latency", "failed_llm_calls", "json_validation_errors"}:
            return ComponentType.MODEL_RESPONSE
        if rule_ids & {"high_tool_latency", "failed_tool_calls"}:
            return ComponentType.PIPELINE
        if rule_ids & {"empty_transcript", "very_low_transcript_activity"}:
            return ComponentType.DATA
        return ComponentType.UNKNOWN

    def _severity(self, evidence: Sequence[EvidenceRecord]) -> Severity:
        """Return the highest observed evidence severity, or low when absent."""
        order = (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
        observed = {record.severity for record in evidence if record.severity}
        return max(observed, key=order.index) if observed else Severity.LOW

    def _evidence_text(
        self,
        evidence: Sequence[EvidenceRecord],
        confidence: ConfidenceResult,
    ) -> List[str]:
        """Return schema-supported evidence text including confidence context."""
        items = [record.explanation or record.description for record in evidence]
        return items + [f"Confidence summary: {confidence.explanation}"]


class LLMReasoner:
    """Coordinate prompt construction and an injected diagnosis provider."""

    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        provider: ReasoningProvider | None = None,
    ) -> None:
        """Initialize dependencies with deterministic defaults."""
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._provider = provider or TemplateReasoningProvider()

    def reason(
        self,
        rules: List[RuleEvaluation],
        evidence: List[EvidenceRecord],
        similar_sessions: List[SimilarSession],
        confidence: ConfidenceResult,
    ) -> FailureDiagnosis:
        """Generate a structured diagnosis through the configured provider."""
        prompt = self._prompt_builder.build(
            rules, evidence, similar_sessions, confidence
        )
        return self._provider.generate(prompt)
