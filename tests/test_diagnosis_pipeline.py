"""Integration tests for end-to-end backend diagnosis orchestration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pipeline.diagnosis_pipeline import DiagnosisPipeline, DiagnosisResult
from pipeline.schema import (
    CallStatus,
    ErrorType,
    EventType,
    InterviewSession,
    LLMCall,
    SessionStatus,
    Speaker,
    TimelineEvent,
    ToolCall,
    TranscriptTurn,
)


def _event(event_id: str, seconds: int, event_type: EventType, text: str) -> TimelineEvent:
    """Create a deterministic timeline event for integration tests."""
    return TimelineEvent(
        event_id=event_id,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        event_type=event_type,
        description=text,
    )


def _session(
    llm_calls: list[LLMCall] | None = None,
    tool_calls: list[ToolCall] | None = None,
    timeline: list[TimelineEvent] | None = None,
) -> InterviewSession:
    """Create a minimal session for pipeline orchestration tests."""
    return InterviewSession(
        session_id=uuid4(), status=SessionStatus.PASSED,
        transcript=[
            TranscriptTurn(turn_id=1, speaker=Speaker.INTERVIEWER, text="Question", tokens=1),
            TranscriptTurn(turn_id=2, speaker=Speaker.CANDIDATE, text="Answer", tokens=1),
            TranscriptTurn(turn_id=3, speaker=Speaker.INTERVIEWER, text="Question", tokens=1),
            TranscriptTurn(turn_id=4, speaker=Speaker.CANDIDATE, text="Answer", tokens=1),
        ],
        llm_calls=llm_calls or [], tool_calls=tool_calls or [],
        timeline=timeline or [_event("start", 0, EventType.TURN, "Started.")],
    )


def _assert_complete(result: DiagnosisResult) -> None:
    """Assert that each backend stage supplied a result collection."""
    assert isinstance(result, DiagnosisResult)
    assert result.diagnosis is not None
    assert result.confidence is not None
    assert len(result.rules) == 10
    assert len(result.signals) == 31


def test_healthy_interview_runs_all_pipeline_stages() -> None:
    """A healthy session produces a complete deterministic aggregate."""
    session = _session()
    pipeline = DiagnosisPipeline()

    result = pipeline.run(session)

    _assert_complete(result)
    assert not result.evidence
    assert result.diagnosis.rules_triggered == []


def test_llm_timeout_session_runs_all_pipeline_stages() -> None:
    """LLM timeout telemetry flows through rules, evidence, and diagnosis."""
    session = _session(
        llm_calls=[
            LLMCall(
                call_id="llm", model_name="test", latency_ms=10_000.0,
                duration_ms=10_000.0, status=CallStatus.TIMEOUT,
                error_type=ErrorType.TIMEOUT,
            )
        ],
        timeline=[_event("timeout", 0, EventType.ERROR, "Request timeout.")],
    )

    result = DiagnosisPipeline().run(session)

    _assert_complete(result)
    assert "high_llm_latency" in result.diagnosis.rules_triggered
    assert any(record.rule_id == "high_llm_latency" for record in result.evidence)


def test_tool_timeout_session_runs_all_pipeline_stages() -> None:
    """Tool timeout telemetry is orchestrated without special pipeline logic."""
    session = _session(
        tool_calls=[
            ToolCall(
                call_id="tool", tool_name="retrieval", duration_ms=10_000.0,
                status=CallStatus.TIMEOUT, error_type=ErrorType.TIMEOUT,
            )
        ]
    )

    result = DiagnosisPipeline().run(session)

    _assert_complete(result)
    assert "high_tool_latency" in result.diagnosis.rules_triggered
    assert result.evidence


def test_mixed_failure_pipeline_is_deterministic() -> None:
    """Repeated orchestration of identical inputs produces equivalent results."""
    session = _session(
        llm_calls=[
            LLMCall(
                call_id="llm", model_name="test", latency_ms=10_000.0,
                status=CallStatus.ERROR, error_type=ErrorType.VALIDATION,
            )
        ],
        tool_calls=[
            ToolCall(
                call_id="tool", tool_name="retrieval", duration_ms=10_000.0,
                status=CallStatus.TIMEOUT, error_type=ErrorType.TIMEOUT,
            )
        ],
        timeline=[
            _event("retry_1", 0, EventType.LLM_CALL, "Retry attempt 1."),
            _event("retry_2", 1, EventType.TOOL_CALL, "Retry attempt 2."),
        ],
    )
    pipeline = DiagnosisPipeline()

    first = pipeline.run(session)
    second = pipeline.run(session)

    _assert_complete(first)
    assert first == second
    assert {"multiple_retries", "json_validation_errors"} <= set(
        first.diagnosis.rules_triggered
    )
