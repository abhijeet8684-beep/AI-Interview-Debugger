"""Tests for FAISS-backed historical session retrieval."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pipeline.schema import (
    EventType,
    ExtractedSignal,
    InterviewSession,
    SessionStatus,
    TimelineEvent,
)
from pipeline.signal_extractor import SignalExtractor
from pipeline.similarity_engine import SimilarityEngine


def _session(latency: float, tokens: int = 500) -> InterviewSession:
    """Create a compact session with deterministic timeline-based features."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tool_latency = 100.0 if latency == 1_000.0 else latency
    return InterviewSession(
        session_id=uuid4(), status=SessionStatus.PASSED,
        timeline=[
            TimelineEvent(event_id="start", timestamp=start, event_type=EventType.TURN),
            TimelineEvent(event_id="end", timestamp=start + timedelta(seconds=latency), event_type=EventType.TURN),
        ],
        extracted_signals=[
            ExtractedSignal(signal_name="average_llm_latency_ms", value=latency, unit="ms", source="test"),
            ExtractedSignal(signal_name="maximum_llm_latency_ms", value=latency, unit="ms", source="test"),
            ExtractedSignal(signal_name="maximum_tool_latency", value=tool_latency, unit="ms", source="test"),
            ExtractedSignal(signal_name="total_tokens", value=tokens, unit="tokens", source="test"),
        ],
        metadata={"test_latency": latency, "test_tokens": tokens},
    )


def _signals(session: InterviewSession, latency: float, tokens: int = 500):
    """Provide stable feature signals without changing the indexed session."""
    signals = SignalExtractor().extract(session)
    values = {signal.signal_name: signal for signal in signals}
    values["average_llm_latency_ms"] = values["average_llm_latency_ms"].model_copy(update={"value": latency})
    values["maximum_llm_latency_ms"] = values["maximum_llm_latency_ms"].model_copy(update={"value": latency})
    values["maximum_tool_latency"] = values["maximum_tool_latency"].model_copy(
        update={"value": 100.0 if latency == 1_000.0 else latency}
    )
    values["total_tokens"] = values["total_tokens"].model_copy(update={"value": tokens})
    return list(values.values())


def test_empty_dataset_returns_no_results() -> None:
    """Searching an empty index returns a sensible empty result set."""
    session = _session(100.0)
    engine = SimilarityEngine()
    engine.build_index([])

    assert engine.search(session, _signals(session, 100.0), [], top_k=5) == []


def test_index_retrieval_is_deterministic_and_sorted() -> None:
    """Indexed sessions are returned in descending, repeatable similarity order."""
    sessions = [_session(100.0), _session(1_000.0), _session(10_000.0)]
    engine = SimilarityEngine()
    engine.build_index(sessions)
    query_signals = _signals(sessions[1], 1_000.0)

    first = engine.search(sessions[1], query_signals, [], top_k=5)
    second = engine.search(sessions[1], query_signals, [], top_k=5)

    assert len(first) == 3
    assert [item.session_id for item in first] == [item.session_id for item in second]
    assert first[0].session_id == sessions[1].session_id
    assert [item.similarity_score for item in first] == sorted(
        (item.similarity_score for item in first), reverse=True
    )


def test_duplicate_sessions_and_large_top_k_are_handled() -> None:
    """Duplicate records and oversized top-k values do not cause failures."""
    session = _session(500.0)
    engine = SimilarityEngine()
    engine.build_index([session, session])

    results = engine.search(session, _signals(session, 500.0), [], top_k=10)

    assert len(results) == 2
    assert all(item.session_id == session.session_id for item in results)


def test_feature_vector_dimension_is_constant() -> None:
    """Feature vectors preserve their dimension despite different sessions."""
    engine = SimilarityEngine()
    first = _session(100.0)
    second = _session(2_000.0)

    assert engine._build_feature_vector(first, _signals(first, 100.0)).shape == (14,)
    assert engine._build_feature_vector(second, _signals(second, 2_000.0)).shape == (14,)
