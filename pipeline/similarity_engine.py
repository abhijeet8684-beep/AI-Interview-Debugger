"""FAISS-backed retrieval of historically similar interview sessions."""
from __future__ import annotations

from typing import List, Sequence

import faiss
import numpy as np

from pipeline.schema import EvidenceRecord, ExtractedSignal, InterviewSession, SimilarSession
from pipeline.signal_extractor import SignalExtractor


class SimilarityEngine:
    """Build and query a reusable normalized numerical FAISS index."""

    def __init__(self) -> None:
        """Initialize an empty, reusable similarity index."""
        self._index: faiss.Index | None = None
        self._sessions: List[InterviewSession] = []
        self._means: np.ndarray | None = None
        self._scales: np.ndarray | None = None

    def build_index(self, sessions: List[InterviewSession]) -> None:
        """Build or replace the historical-session FAISS index.

        Args:
            sessions: Historical sessions to index. Duplicate records are
                retained because they are valid historical observations.
        """
        self._sessions = list(sessions)
        if not self._sessions:
            self._index = None
            self._means = None
            self._scales = None
            return

        vectors = np.vstack([self._historical_vector(session) for session in sessions])
        self._means = vectors.mean(axis=0)
        scales = vectors.std(axis=0)
        self._scales = np.where(scales == 0.0, 1.0, scales)
        normalized_vectors = self._normalize(vectors)
        self._index = faiss.IndexFlatIP(normalized_vectors.shape[1])
        self._index.add(normalized_vectors)

    def search(
        self,
        session: InterviewSession,
        signals: List[ExtractedSignal],
        evidence: List[EvidenceRecord],
        top_k: int = 5,
    ) -> List[SimilarSession]:
        """Return the top-k historical sessions by cosine similarity.

        Args:
            session: Current session being compared with indexed history.
            signals: Already extracted current-session signals.
            evidence: Current-session evidence. It is accepted to preserve the
                pipeline contract; retrieval remains feature-vector based.
            top_k: Maximum number of historical results to return.

        Returns:
            Similar sessions sorted by descending cosine similarity.
        """
        del evidence
        if self._index is None or top_k <= 0:
            return []
        query = self._normalize(self._build_feature_vector(session, signals))
        limit = min(top_k, len(self._sessions))
        scores, indices = self._index.search(query, limit)
        return [
            self._similar_session(self._sessions[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def _historical_vector(self, session: InterviewSession) -> np.ndarray:
        """Build features for an indexed session using available signals."""
        signals = session.extracted_signals or SignalExtractor().extract(session)
        return self._build_feature_vector(session, signals)

    def _build_feature_vector(
        self, session: InterviewSession, signals: Sequence[ExtractedSignal]
    ) -> np.ndarray:
        """Build one fixed-order numerical feature vector from session signals."""
        del session
        values = {signal.signal_name: signal.value for signal in signals}
        feature_names = (
            "average_llm_latency_ms", "maximum_llm_latency_ms",
            "average_tool_latency", "maximum_tool_latency", "failed_llm_calls",
            "failed_tool_calls", "retry_count", "transcript_turn_count",
            "interview_duration_seconds", "total_tokens", "high_latency_detected",
            "empty_transcript", "multiple_retries", "tool_failures_detected",
        )
        return np.asarray([values.get(name, 0.0) for name in feature_names], dtype=np.float32)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Standardize vectors with indexed statistics and apply L2 normalization."""
        if self._means is None or self._scales is None:
            return vectors.reshape(1, -1).astype(np.float32)
        normalized = (np.atleast_2d(vectors) - self._means) / self._scales
        normalized = normalized.astype(np.float32)
        faiss.normalize_L2(normalized)
        return normalized

    def _similar_session(
        self, session: InterviewSession, score: float
    ) -> SimilarSession:
        """Convert one historical record into the frozen similarity schema."""
        metrics = session.metrics
        diagnosis = session.diagnosis
        ground_truth = session.ground_truth
        failure_type = (
            ground_truth.expected_failure_type if ground_truth else None
        )
        duration = self._duration_seconds(session)
        summary = (
            f"duration_seconds={duration:.2f}; "
            f"average_llm_latency_ms={metrics.avg_llm_latency_ms if metrics else None}; "
            f"average_tool_latency_ms={self._average_tool_latency(session):.2f}; "
            f"supporting_rule_count={len(diagnosis.rules_triggered) if diagnosis else 0}; "
            f"dataset_version={session.dataset_version}; "
            f"pipeline_version={session.pipeline_version}"
        )
        return SimilarSession(
            session_id=session.session_id,
            similarity_score=max(0.0, min(1.0, score)),
            failure_type=failure_type,
            summary=summary,
        )

    def _duration_seconds(self, session: InterviewSession) -> float:
        """Calculate timeline duration when at least two timestamps exist."""
        timestamps = [event.timestamp for event in session.timeline if event.timestamp]
        return (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0.0

    def _average_tool_latency(self, session: InterviewSession) -> float:
        """Calculate average tool duration from available historical calls."""
        durations = [call.duration_ms for call in session.tool_calls if call.duration_ms is not None]
        return sum(durations) / len(durations) if durations else 0.0
