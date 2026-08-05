"""Extract measurable, structured signals from interview sessions.

This module deliberately contains no diagnosis, rule evaluation, or model
inference. It transforms observable session records into ``ExtractedSignal``
objects for later pipeline stages.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Sequence

from pipeline.schema import (
    CallStatus,
    EventType,
    ExtractedSignal,
    InterviewSession,
    Speaker,
    ToolCall,
    TranscriptTurn,
)


class SignalExtractor:
    """Extract deterministic, measurable signals from an interview session."""

    def extract(self, session: InterviewSession) -> List[ExtractedSignal]:
        """Extract all supported signal categories from a session.

        Args:
            session: Interview session containing raw transcript, call, and
                timeline records.

        Returns:
            A flat list of structured numerical signals.
        """
        signals: List[ExtractedSignal] = []
        signals.extend(self._extract_interview_signals(session))
        signals.extend(self._extract_llm_signals(session))
        signals.extend(self._extract_tool_signals(session))
        signals.extend(self._extract_timeline_signals(session))
        signals.extend(self._extract_transcript_signals(session))
        signals.extend(self._extract_failure_indicators(session))
        return signals

    def _extract_interview_signals(
        self, session: InterviewSession
    ) -> List[ExtractedSignal]:
        """Extract session duration, turn counts, and observed stage count."""
        timestamps = [
            event.timestamp for event in session.timeline if event.timestamp is not None
        ]
        stages = {
            item.stage
            for item in [*session.transcript, *session.timeline]
            if item.stage is not None
        }
        interviewer_turns = self._turns_for_speaker(session.transcript, Speaker.INTERVIEWER)
        candidate_turns = self._turns_for_speaker(session.transcript, Speaker.CANDIDATE)
        return [
            self._signal(
                "interview_duration_seconds",
                self._duration_seconds(timestamps),
                "seconds",
                "timeline",
            ),
            self._signal("transcript_turn_count", len(session.transcript), "turns", "transcript"),
            self._signal("interviewer_turn_count", len(interviewer_turns), "turns", "transcript"),
            self._signal("candidate_turn_count", len(candidate_turns), "turns", "transcript"),
            self._signal("interview_stage_count", len(stages), "stages", "session"),
        ]

    def _extract_llm_signals(self, session: InterviewSession) -> List[ExtractedSignal]:
        """Extract LLM-call volumes, statuses, timing, and token metrics."""
        calls = session.llm_calls
        latencies = [call.latency_ms for call in calls if call.latency_ms is not None]
        response_times = [
            call.duration_ms if call.duration_ms is not None else call.latency_ms
            for call in calls
            if call.duration_ms is not None or call.latency_ms is not None
        ]
        return [
            self._signal("total_llm_calls", len(calls), "calls", "llm_calls"),
            self._signal(
                "failed_llm_calls",
                sum(self._is_failed_call(call.status) for call in calls),
                "calls",
                "llm_calls",
            ),
            self._signal("average_llm_latency_ms", self._average(latencies), "milliseconds", "llm_calls"),
            self._signal("maximum_llm_latency_ms", self._maximum(latencies), "milliseconds", "llm_calls"),
            self._signal(
                "total_prompt_tokens",
                sum(call.tokens_input or 0 for call in calls),
                "tokens",
                "llm_calls",
            ),
            self._signal(
                "total_completion_tokens",
                sum(call.tokens_output or 0 for call in calls),
                "tokens",
                "llm_calls",
            ),
            self._signal(
                "total_tokens",
                sum((call.tokens_input or 0) + (call.tokens_output or 0) for call in calls),
                "tokens",
                "llm_calls",
            ),
            self._signal("average_response_time", self._average(response_times), "milliseconds", "llm_calls"),
        ]

    def _extract_tool_signals(self, session: InterviewSession) -> List[ExtractedSignal]:
        """Extract tool-call volumes, statuses, latency, and retry metrics."""
        calls = session.tool_calls
        durations = [call.duration_ms for call in calls if call.duration_ms is not None]
        return [
            self._signal("total_tool_calls", len(calls), "calls", "tool_calls"),
            self._signal(
                "failed_tool_calls",
                sum(self._is_failed_call(call.status) for call in calls),
                "calls",
                "tool_calls",
            ),
            self._signal("average_tool_latency", self._average(durations), "milliseconds", "tool_calls"),
            self._signal("maximum_tool_latency", self._maximum(durations), "milliseconds", "tool_calls"),
            self._signal("retry_count", self._retry_count(session), "retries", "timeline"),
        ]

    def _extract_timeline_signals(
        self, session: InterviewSession
    ) -> List[ExtractedSignal]:
        """Extract event counts and observable timeout/retry/failure events."""
        descriptions = [event.description or "" for event in session.timeline]
        return [
            self._signal("total_events", len(session.timeline), "events", "timeline"),
            self._signal(
                "timeout_events",
                sum("timeout" in description.lower() for description in descriptions),
                "events",
                "timeline",
            ),
            self._signal("retry_events", self._retry_count(session), "events", "timeline"),
            self._signal(
                "failure_events",
                sum(event.event_type == EventType.ERROR for event in session.timeline),
                "events",
                "timeline",
            ),
        ]

    def _extract_transcript_signals(
        self, session: InterviewSession
    ) -> List[ExtractedSignal]:
        """Extract response lengths and empty-turn counts from the transcript."""
        candidate_turns = self._turns_for_speaker(session.transcript, Speaker.CANDIDATE)
        interviewer_turns = self._turns_for_speaker(session.transcript, Speaker.INTERVIEWER)
        empty_turns = [turn for turn in session.transcript if not turn.text.strip()]
        return [
            self._signal(
                "average_candidate_response_length",
                self._average([self._turn_length(turn) for turn in candidate_turns]),
                "tokens",
                "transcript",
            ),
            self._signal(
                "average_interviewer_question_length",
                self._average([self._turn_length(turn) for turn in interviewer_turns]),
                "tokens",
                "transcript",
            ),
            self._signal("empty_candidate_responses", sum(not turn.text.strip() for turn in candidate_turns), "turns", "transcript"),
            self._signal("empty_transcript_turns", len(empty_turns), "turns", "transcript"),
        ]

    def _extract_failure_indicators(
        self, session: InterviewSession
    ) -> List[ExtractedSignal]:
        """Extract measurable indicator flags without assigning a root cause."""
        llm_latencies = [call.latency_ms or 0.0 for call in session.llm_calls]
        tool_durations = [call.duration_ms or 0.0 for call in session.tool_calls]
        error_messages = [
            call.error_message or ""
            for call in [*session.llm_calls, *session.tool_calls]
        ]
        has_validation_error = any(
            getattr(call.error_type, "value", call.error_type) == "validation"
            for call in [*session.llm_calls, *session.tool_calls]
        )
        empty_transcript = not session.transcript or any(
            not turn.text.strip() for turn in session.transcript
        )
        failed_tools = any(
            self._is_failed_call(call.status) for call in session.tool_calls
        )
        return [
            self._signal(
                "high_latency_detected",
                int(max([*llm_latencies, *tool_durations], default=0.0) >= 5000.0),
                "boolean",
                "calls",
            ),
            self._signal(
                "multiple_retries",
                int(self._retry_count(session) >= 2),
                "boolean",
                "timeline",
            ),
            self._signal("empty_transcript", int(empty_transcript), "boolean", "transcript"),
            self._signal(
                "json_errors_detected",
                int(has_validation_error or any("json" in message.lower() for message in error_messages)),
                "boolean",
                "calls",
            ),
            self._signal("tool_failures_detected", int(failed_tools), "boolean", "tool_calls"),
        ]

    def _signal(
        self, name: str, value: float | int, unit: str, source: str
    ) -> ExtractedSignal:
        """Create one fully observed signal with deterministic confidence."""
        return ExtractedSignal(
            signal_name=name,
            value=float(value),
            unit=unit,
            source=source,
            confidence=1.0,
        )

    def _turns_for_speaker(
        self, turns: Sequence[TranscriptTurn], speaker: Speaker
    ) -> List[TranscriptTurn]:
        """Return transcript turns for a specific speaker role."""
        return [turn for turn in turns if turn.speaker == speaker]

    def _duration_seconds(self, timestamps: Sequence[datetime]) -> float:
        """Return elapsed seconds between the earliest and latest timestamps."""
        if len(timestamps) < 2:
            return 0.0
        return (max(timestamps) - min(timestamps)).total_seconds()

    def _average(self, values: Iterable[float | int]) -> float:
        """Return a rounded average, or zero for an empty collection."""
        items = list(values)
        return round(sum(items) / len(items), 2) if items else 0.0

    def _maximum(self, values: Iterable[float | int]) -> float:
        """Return the maximum value, or zero for an empty collection."""
        items = list(values)
        return float(max(items)) if items else 0.0

    def _turn_length(self, turn: TranscriptTurn) -> int:
        """Return supplied token count, with a deterministic text fallback."""
        return turn.tokens if turn.tokens is not None else len(turn.text.split())

    def _retry_count(self, session: InterviewSession) -> int:
        """Count retry events recorded in the session timeline."""
        return sum(
            "retry" in (event.description or "").lower()
            for event in session.timeline
        )

    def _is_successful_call(self, status: object) -> bool:
        """Treat standardized and legacy successful statuses as successful."""
        normalized = getattr(status, "value", status)
        return normalized in (CallStatus.SUCCESS.value, CallStatus.OK.value)

    def _is_failed_call(self, status: object) -> bool:
        """Return whether a call has an explicit non-success execution status."""
        return status is not None and not self._is_successful_call(status)
