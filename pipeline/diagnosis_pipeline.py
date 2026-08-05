"""Orchestrate the deterministic AI Interview Debugger backend pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from pipeline.confidence_engine import ConfidenceEngine, ConfidenceResult
from pipeline.evidence_builder import EvidenceBuilder
from pipeline.llm_reasoner import LLMReasoner
from pipeline.rule_engine import RuleEngine
from pipeline.schema import (
    EvidenceRecord,
    ExtractedSignal,
    FailureDiagnosis,
    InterviewSession,
    RuleEvaluation,
    SimilarSession,
)
from pipeline.signal_extractor import SignalExtractor
from pipeline.similarity_engine import SimilarityEngine


@dataclass(frozen=True)
class DiagnosisResult:
    """Immutable aggregate of outputs produced by the backend pipeline stages."""

    diagnosis: FailureDiagnosis
    confidence: ConfidenceResult
    evidence: Tuple[EvidenceRecord, ...]
    rules: Tuple[RuleEvaluation, ...]
    similar_sessions: Tuple[SimilarSession, ...]
    signals: Tuple[ExtractedSignal, ...]


class DiagnosisPipeline:
    """Coordinate independently testable backend analysis stages."""

    def __init__(
        self,
        historical_sessions: Sequence[InterviewSession] | None = None,
        signal_extractor: SignalExtractor | None = None,
        rule_engine: RuleEngine | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        similarity_engine: SimilarityEngine | None = None,
        confidence_engine: ConfidenceEngine | None = None,
        reasoner: LLMReasoner | None = None,
    ) -> None:
        """Initialize pipeline stages with injectable or default implementations."""
        self._signal_extractor = signal_extractor or SignalExtractor()
        self._rule_engine = rule_engine or RuleEngine()
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        self._similarity_engine = similarity_engine or SimilarityEngine()
        self._confidence_engine = confidence_engine or ConfidenceEngine()
        self._reasoner = reasoner or LLMReasoner()
        if historical_sessions is not None:
            self._similarity_engine.build_index(list(historical_sessions))

    def run(self, session: InterviewSession) -> DiagnosisResult:
        """Execute each frozen backend stage for one interview session.

        Args:
            session: Session to analyze without mutating its contents.

        Returns:
            Immutable aggregate of all stage outputs.
        """
        signals = self._signal_extractor.extract(session)
        rules = self._rule_engine.evaluate(signals)
        evidence = self._evidence_builder.build(signals, rules)
        similar_sessions = self._similarity_engine.search(session, signals, evidence)
        confidence = self._confidence_engine.calculate(
            rules, evidence, similar_sessions
        )
        diagnosis = self._reasoner.reason(
            rules, evidence, similar_sessions, confidence
        )
        return DiagnosisResult(
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=tuple(evidence),
            rules=tuple(rules),
            similar_sessions=tuple(similar_sessions),
            signals=tuple(signals),
        )
