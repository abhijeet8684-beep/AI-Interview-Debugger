"""Tests for deterministic signal extraction from interview sessions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from pipeline.schema import (
    CallStatus,
    ErrorType,
    EventType,
    InterviewSession,
    InterviewStage,
    LLMCall,
    SessionStatus,
    Speaker,
    TimelineEvent,
    ToolCall,
    TranscriptTurn,
)
from pipeline.signal_extractor import SignalExtractor


def _signal_values(session: InterviewSession) -> dict[str, float]:
    """Extract a signal-name-to-value mapping for concise assertions."""
    return {
        signal.signal_name: signal.value
        for signal in SignalExtractor().extract(session)
    }


def _session(
    transcript: list[TranscriptTurn] | None = None,
    llm_calls: list[LLMCall] | None = None,
    tool_calls: list[ToolCall] | None = None,
    timeline: list[TimelineEvent] | None = None,
) -> InterviewSession:
    """Build a minimal schema-valid session for extractor tests."""
    return InterviewSession(
        session_id=uuid4(),
        status=SessionStatus.PASSED,
        transcript=transcript or [],
        llm_calls=llm_calls or [],
        tool_calls=tool_calls or [],
        timeline=timeline or [],
    )


def _event(
    event_id: str,
    timestamp: datetime,
    event_type: EventType,
    description: str,
) -> TimelineEvent:
    """Create a compact timeline event with an interview stage."""
    return TimelineEvent(
        event_id=event_id,
        timestamp=timestamp,
        event_type=event_type,
        description=description,
        stage=InterviewStage.TECHNICAL,
    )


def test_successful_interview_signals_are_correct() -> None:
    """Successful records yield accurate metrics and no failure indicators."""
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = _session(
        transcript=[
            TranscriptTurn(turn_id=1, speaker=Speaker.INTERVIEWER, text="Explain cross validation.", tokens=4),
            TranscriptTurn(turn_id=2, speaker=Speaker.CANDIDATE, text="I would split the data into folds.", tokens=8),
        ],
        llm_calls=[
            LLMCall(
                call_id="llm_1", model_name="test-model", latency_ms=500.0,
                duration_ms=600.0, status=CallStatus.SUCCESS, tokens_input=100,
                tokens_output=20,
            )
        ],
        tool_calls=[
            ToolCall(
                call_id="tool_1", tool_name="retrieval", duration_ms=200.0,
                status=CallStatus.SUCCESS,
            )
        ],
        timeline=[
            _event("event_1", started, EventType.TURN, "Interview started."),
            _event("event_2", started + timedelta(seconds=30), EventType.LLM_CALL, "Evaluation completed."),
        ],
    )

    values = _signal_values(session)

    assert values["interview_duration_seconds"] == pytest.approx(30.0)
    assert values["transcript_turn_count"] == 2.0
    assert values["candidate_turn_count"] == 1.0
    assert values["total_llm_calls"] == 1.0
    assert values["total_tokens"] == 120.0
    assert values["average_tool_latency"] == pytest.approx(200.0)
    assert values["high_latency_detected"] == 0.0
    assert values["tool_failures_detected"] == 0.0
    assert len(SignalExtractor().extract(session)) == 31


def test_llm_timeout_session_signals_are_correct() -> None:
    """LLM timeout telemetry is represented without diagnosing its cause."""
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = _session(
        llm_calls=[
            LLMCall(
                call_id="llm_timeout", model_name="test-model", latency_ms=31_000.0,
                duration_ms=31_000.0, status=CallStatus.TIMEOUT,
                error_type=ErrorType.TIMEOUT, error_message="Request timeout.",
            )
        ],
        timeline=[
            _event("event_1", started, EventType.LLM_CALL, "Request timeout."),
            _event("event_2", started + timedelta(seconds=2), EventType.ERROR, "Request timeout."),
            _event("event_3", started + timedelta(seconds=4), EventType.LLM_CALL, "Retry attempt 1 started."),
            _event("event_4", started + timedelta(seconds=6), EventType.LLM_CALL, "Retry attempt 2 started."),
        ],
    )

    values = _signal_values(session)

    assert values["failed_llm_calls"] == 1.0
    assert values["maximum_llm_latency_ms"] == pytest.approx(31_000.0)
    assert values["timeout_events"] == 2.0
    assert values["retry_count"] == 2.0
    assert values["high_latency_detected"] == 1.0
    assert values["multiple_retries"] == 1.0


def test_speech_to_text_failure_transcript_is_measurable() -> None:
    """Low-confidence transcript content remains extractable without errors."""
    session = _session(
        transcript=[
            TranscriptTurn(turn_id=1, speaker=Speaker.INTERVIEWER, text="Describe your project.", tokens=3),
            TranscriptTurn(
                turn_id=2,
                speaker=Speaker.CANDIDATE,
                text="[inaudible] model training [uncertain transcription]",
                tokens=5,
                annotations={"speech_confidence": 0.34},
            ),
        ]
    )

    values = _signal_values(session)

    assert values["average_candidate_response_length"] == pytest.approx(5.0)
    assert values["empty_candidate_responses"] == 0.0
    assert values["empty_transcript"] == 0.0


def test_tool_timeout_signals_are_correct() -> None:
    """Tool timeout telemetry produces observable tool and latency indicators."""
    session = _session(
        tool_calls=[
            ToolCall(
                call_id="tool_timeout", tool_name="retrieval", duration_ms=20_000.0,
                status=CallStatus.TIMEOUT, error_type=ErrorType.TIMEOUT,
                error_message="Retrieval timeout.",
            )
        ]
    )

    values = _signal_values(session)

    assert values["failed_tool_calls"] == 1.0
    assert values["maximum_tool_latency"] == pytest.approx(20_000.0)
    assert values["tool_failures_detected"] == 1.0
    assert values["high_latency_detected"] == 1.0


def test_empty_transcript_returns_signals_without_exceptions() -> None:
    """An empty transcript is handled as an observable condition."""
    values = _signal_values(_session())

    assert values["transcript_turn_count"] == 0.0
    assert values["empty_transcript_turns"] == 0.0
    assert values["empty_transcript"] == 1.0
    assert values["average_candidate_response_length"] == 0.0


def test_mixed_failure_session_produces_all_expected_indicators() -> None:
    """Combined telemetry is extracted without assigning a failure diagnosis."""
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session = _session(
        llm_calls=[
            LLMCall(
                call_id="llm_timeout",
                model_name="test-model",
                latency_ms=30_500.5,
                duration_ms=30_500.5,
                status=CallStatus.TIMEOUT,
                error_type=ErrorType.TIMEOUT,
                error_message="LLM request timeout.",
            ),
            LLMCall(
                call_id="llm_json",
                model_name="test-model",
                latency_ms=750.0,
                status=CallStatus.ERROR,
                error_type=ErrorType.VALIDATION,
                error_message="Invalid JSON evaluation payload.",
            ),
        ],
        tool_calls=[
            ToolCall(
                call_id="tool_timeout",
                tool_name="retrieval",
                duration_ms=18_000.0,
                status=CallStatus.TIMEOUT,
                error_type=ErrorType.TIMEOUT,
                error_message="Retrieval timeout.",
            )
        ],
        timeline=[
            _event("event_1", started, EventType.ERROR, "LLM request timeout."),
            _event("event_2", started + timedelta(seconds=1), EventType.ERROR, "Retrieval timeout."),
            _event("event_3", started + timedelta(seconds=2), EventType.LLM_CALL, "Retry attempt 1 started."),
            _event("event_4", started + timedelta(seconds=3), EventType.TOOL_CALL, "Retry attempt 2 started."),
        ],
    )

    values = _signal_values(session)

    assert values["failed_llm_calls"] == 2.0
    assert values["failed_tool_calls"] == 1.0
    assert values["timeout_events"] == 2.0
    assert values["retry_events"] == 2.0
    assert values["failure_events"] == 2.0
    assert values["high_latency_detected"] == 1.0
    assert values["multiple_retries"] == 1.0
    assert values["json_errors_detected"] == 1.0
    assert values["tool_failures_detected"] == 1.0
    assert values["average_llm_latency_ms"] == pytest.approx(15_625.25)


def test_large_session_extracts_expected_signals_without_exceptions() -> None:
    """A large session remains extractable and retains the complete contract."""
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    transcript = [
        TranscriptTurn(
            turn_id=index,
            speaker=Speaker.INTERVIEWER if index % 2 else Speaker.CANDIDATE,
            text=f"Transcript turn {index}",
            tokens=4 + (index % 5),
        )
        for index in range(1, 201)
    ]
    llm_calls = [
        LLMCall(
            call_id=f"llm_{index}",
            model_name="test-model",
            latency_ms=300.0 + index,
            duration_ms=350.0 + index,
            status=CallStatus.SUCCESS,
            tokens_input=100,
            tokens_output=25,
        )
        for index in range(100)
    ]
    tool_calls = [
        ToolCall(
            call_id=f"tool_{index}",
            tool_name="retrieval",
            duration_ms=80.0 + index,
            status=CallStatus.SUCCESS,
        )
        for index in range(100)
    ]
    timeline = [
        _event(
            f"event_{index}",
            started + timedelta(seconds=index),
            EventType.TURN,
            "Transcript event.",
        )
        for index in range(300)
    ]

    signals = SignalExtractor().extract(
        _session(transcript=transcript, llm_calls=llm_calls, tool_calls=tool_calls, timeline=timeline)
    )
    names = {signal.signal_name for signal in signals}

    assert len(signals) == 31
    assert {"total_llm_calls", "total_tool_calls", "total_events", "total_tokens"} <= names
