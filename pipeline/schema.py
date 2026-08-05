"""Pydantic schema models for AI-Interview-Debugger pipeline.

This module defines strongly-typed, nested Pydantic models used across the
pipeline to represent synthetic interview sessions, calls to LLMs/tools, timeline
events, session-level metrics, and the failure diagnosis produced by the
system.

Design notes
- Models are intentionally passive (no business logic or validation-heavy
  methods) so they remain easy to serialize and use in tests.
- Use enums for common categorical fields to improve type-safety and
  downstream filtering.
- Keep fields optional where data may be missing (e.g., diagnosis before
  pipeline runs).

Example JSON (realistic single-session object)
----------------------------------------------
{
  "session_id": "7f9b6d2a-3c1b-4f2e-8d2a-6a3f2d5f1a9b",
  "status": "failed",
  "created_at": "2026-08-05T14:23:00Z",
  "candidate": {
    "candidate_id": "cand_12345",
    "name": "Alice Doe",
    "experience_years": 2.5,
    "metadata": {"role": "ML Intern", "location": "Remote"}
  },
  "transcript": [
    {"turn_id": 1, "speaker": "interviewer", "text": "Please explain logistic regression.", "timestamp": "2026-08-05T14:24:00Z"},
    {"turn_id": 2, "speaker": "candidate", "text": "It models probability using a sigmoid...", "timestamp": "2026-08-05T14:24:15Z"}
  ],
  "llm_calls": [
    {"call_id": "llm_1", "model_name": "gpt-4o-mini", "prompt": "Explain logistic regression briefly.", "response": "Logistic regression models the probability...", "latency_ms": 420}
  ],
  "tool_calls": [],
  "timeline": [
    {"event_id": "e1", "timestamp": "2026-08-05T14:24:00Z", "event_type": "TURN", "description": "Interviewer asked a question"}
  ],
  "metrics": {"num_turns": 2, "avg_llm_latency_ms": 420.0, "num_errors": 0},
  "diagnosis": {
    "root_cause": "knowledge_gap",
    "affected_component": "model_response",
    "severity": "medium",
    "confidence": 0.72,
    "evidence": ["Candidate used incorrect terminology", "Low similarity to expected answer"],
    "rules_triggered": ["keyword_mismatch"]
  }
}
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SessionStatus(str, Enum):
    """High-level session status."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    RUNNING = "running"


class Severity(str, Enum):
    """Severity levels for detected failures."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentType(str, Enum):
    """Types of components that can be affected in a failure."""

    MODEL_RESPONSE = "model_response"
    PIPELINE = "pipeline"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


class Speaker(str, Enum):
    """Speaker role for transcript turns."""

    CANDIDATE = "candidate"
    INTERVIEWER = "interviewer"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class EventType(str, Enum):
    """Types of timeline events."""

    TURN = "TURN"
    LLM_CALL = "LLM_CALL"
    TOOL_CALL = "TOOL_CALL"
    ERROR = "ERROR"
    METRIC = "METRIC"
    DIAGNOSIS = "DIAGNOSIS"


class FailureType(str, Enum):
    """Categorized root causes for failures. Keep finite but extensible."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    HALLUCINATION = "hallucination"
    KNOWLEDGE_GAP = "knowledge_gap"
    PROMPT_ISSUE = "prompt_issue"
    TOOL_FAILURE = "tool_failure"
    UNKNOWN = "unknown"


class CallStatus(str, Enum):
    """Standardized execution statuses for LLM and tool calls."""

    SUCCESS = "success"
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"


class ErrorType(str, Enum):
    """Normalized operational error categories for external calls."""

    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    PROVIDER = "provider"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class InterviewStage(str, Enum):
    """Interview stages that may be associated with turns and timeline events."""

    INTRODUCTION = "introduction"
    TECHNICAL = "technical"
    CODING = "coding"
    BEHAVIORAL = "behavioral"
    EVALUATION = "evaluation"


class Candidate(BaseModel):
    """Information about the interview candidate.

    Fields:
    - candidate_id: stable id for joining across datasets
    - name: human-readable name (may be synthetic)
    - experience_years: optional float representing years of experience
    - metadata: free-form metadata for experiments (role, location, tags)
    """

    candidate_id: str = Field(..., description="Unique candidate identifier")
    name: Optional[str] = Field(None, description="Candidate name (optional)")
    experience_years: Optional[float] = Field(
        None, description="Years of professional experience (approx.)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Free-form metadata about candidate"
    )

    model_config = ConfigDict(from_attributes=True)


class TranscriptTurn(BaseModel):
    """A single conversational turn in the interview transcript.

    Fields:
    - turn_id: integer turn index (1-based)
    - speaker: Speaker enum
    - text: content of the turn
    - timestamp: ISO-8601 timestamp when the turn occurred
    - tokens: optional token count (if available)
    - annotations: optional structured annotations (entities, intents)
    """

    turn_id: int = Field(..., description="1-based index of the turn in the session")
    trace_id: Optional[str] = Field(
        None,
        description="Correlation identifier shared by work triggered from this turn",
    )
    stage: Optional[InterviewStage] = Field(
        None, description="Interview stage associated with this turn"
    )
    speaker: Speaker = Field(..., description="Who produced this turn")
    text: str = Field(..., description="Text content of the turn")
    timestamp: Optional[datetime] = Field(None, description="When the turn occurred")
    tokens: Optional[int] = Field(None, description="Token count for the turn text")
    annotations: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional structured annotations"
    )

    model_config = ConfigDict(from_attributes=True)


class LLMCall(BaseModel):
    """Record of a call to a language model.

    Fields:
    - call_id: stable identifier for tracing
    - model_name: model identifier (eg. 'gpt-4o-mini')
    - prompt: prompt text sent to the model
    - response: text returned by the model
    - timestamp: when the call was made
    - latency_ms: measured round-trip latency
    - tokens_input / tokens_output: optional token counts
    - metadata: provider-specific metadata (status codes, request ids)
    """

    call_id: str = Field(..., description="Unique LLM call identifier")
    trace_id: Optional[str] = Field(
        None, description="Correlation identifier for the enclosing workflow trace"
    )
    related_turn_id: Optional[int] = Field(
        None, description="Transcript turn that triggered this LLM call"
    )
    parent_call_id: Optional[str] = Field(
        None, description="Optional parent LLM or tool call identifier"
    )
    model_name: str = Field(..., description="Model name or alias")
    prompt: Optional[str] = Field(None, description="Prompt sent to the model")
    response: Optional[str] = Field(None, description="Model's textual response")
    timestamp: Optional[datetime] = Field(None, description="Call time")
    latency_ms: Optional[float] = Field(None, description="Observed latency in ms")
    duration_ms: Optional[float] = Field(
        None, description="Total execution duration in milliseconds"
    )
    status: Optional[CallStatus] = Field(
        None, description="Normalized execution status"
    )
    error_type: Optional[ErrorType] = Field(
        None, description="Normalized category for a failed call"
    )
    error_message: Optional[str] = Field(
        None, description="Safe, human-readable error detail for a failed call"
    )
    tokens_input: Optional[int] = Field(None, description="Input token count")
    tokens_output: Optional[int] = Field(None, description="Output token count")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Provider-specific metadata"
    )

    model_config = ConfigDict(from_attributes=True)


class ToolCall(BaseModel):
    """Record of an external tool or API call used during the session.

    Examples: code-execution, retrieval, knowledge-base lookup.
    """

    call_id: str = Field(..., description="Unique tool call identifier")
    trace_id: Optional[str] = Field(
        None, description="Correlation identifier for the enclosing workflow trace"
    )
    related_turn_id: Optional[int] = Field(
        None, description="Transcript turn that triggered this tool call"
    )
    parent_call_id: Optional[str] = Field(
        None, description="Optional parent LLM or tool call identifier"
    )
    tool_name: str = Field(..., description="Tool or API name")
    input: Optional[Dict[str, Any]] = Field(None, description="Structured input")
    output: Optional[Dict[str, Any]] = Field(None, description="Structured output")
    timestamp: Optional[datetime] = Field(None, description="Call timestamp")
    duration_ms: Optional[float] = Field(
        None, description="Total execution duration in milliseconds"
    )
    status: Optional[Union[CallStatus, str]] = Field(
        None,
        description="Normalized execution status; legacy status strings remain accepted",
    )
    error_type: Optional[ErrorType] = Field(
        None, description="Normalized category for a failed call"
    )
    error_message: Optional[str] = Field(
        None, description="Safe, human-readable error detail for a failed call"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Tool-specific metadata"
    )

    model_config = ConfigDict(from_attributes=True)


class TimelineEvent(BaseModel):
    """An event in the reconstructed session timeline.

    Events should be small, typed records that reference relevant ids
    (turn_id, call_id) where applicable to make reconstruction easy.
    """

    event_id: str = Field(..., description="Unique event identifier")
    trace_id: Optional[str] = Field(
        None, description="Correlation identifier for the enclosing workflow trace"
    )
    stage: Optional[InterviewStage] = Field(
        None, description="Interview stage associated with this event"
    )
    timestamp: Optional[datetime] = Field(None, description="Event timestamp")
    event_type: EventType = Field(..., description="Type of the event")
    description: Optional[str] = Field(None, description="Human-readable description")
    related_turn_id: Optional[int] = Field(None, description="If related to a transcript turn")
    related_call_id: Optional[str] = Field(None, description="If related to an LLM/tool call")
    related_event_id: Optional[str] = Field(
        None, description="Optional causal predecessor event identifier"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Event-specific metadata"
    )

    model_config = ConfigDict(from_attributes=True)


class SessionMetrics(BaseModel):
    """Session-level aggregated metrics useful for diagnostics and ranking.

    Keep metrics compact and numeric to make downstream scoring simple.
    """

    num_turns: Optional[int] = Field(None, description="Total number of turns")
    avg_turn_length: Optional[float] = Field(None, description="Average tokens or chars per turn")
    avg_llm_latency_ms: Optional[float] = Field(None, description="Average LLM latency in ms")
    total_tokens: Optional[int] = Field(None, description="Total tokens used in session")
    num_errors: Optional[int] = Field(None, description="Number of explicit errors observed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extra metrics")

    model_config = ConfigDict(from_attributes=True)


class FailureDiagnosis(BaseModel):
    """Structured diagnosis produced by the pipeline.

    Fields:
    - root_cause: categorized failure type
    - affected_component: which component is implicated
    - severity: human-friendly severity label
    - confidence: [0,1] estimate of confidence
    - evidence: short list of textual evidence strings
    - rules_triggered: identifiers of deterministic rules that fired
    - similar_sessions: optional list of session ids that are similar
    """

    root_cause: FailureType = Field(..., description="Categorized root cause")
    affected_component: ComponentType = Field(..., description="Affected component")
    severity: Severity = Field(..., description="Severity label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence snippets")
    evidence_records: List["EvidenceRecord"] = Field(
        default_factory=list,
        description="Structured evidence with references to supporting records",
    )
    rules_triggered: List[str] = Field(default_factory=list, description="IDs of rules that fired")
    rule_evaluations: List["RuleEvaluation"] = Field(
        default_factory=list,
        description="Structured outcomes of deterministic rule evaluations",
    )
    similar_sessions: Optional[List[UUID]] = Field(
        default=None, description="Optional list of similar historical session IDs"
    )
    similar_session_records: List["SimilarSession"] = Field(
        default_factory=list,
        description="Rich similarity retrieval results with scores and summaries",
    )

    model_config = ConfigDict(from_attributes=True)


class EvidenceRecord(BaseModel):
    """Structured, traceable evidence supporting a failure diagnosis.

    The existing ``FailureDiagnosis.evidence`` field is retained for backwards
    compatibility. New pipeline stages should populate this model when source
    attribution is available.
    """

    evidence_id: str = Field(..., description="Unique evidence identifier")
    description: str = Field(..., description="Human-readable evidence summary")
    source_type: str = Field(
        ..., description="Source record type, such as transcript_turn or llm_call"
    )
    source_id: str = Field(..., description="Identifier of the supporting source record")
    trace_id: Optional[str] = Field(
        None, description="Workflow trace shared with the supporting source"
    )
    excerpt: Optional[str] = Field(None, description="Relevant source excerpt")
    metric: Optional[str] = Field(None, description="Name of the supporting metric")
    observed_value: Optional[float] = Field(
        None, description="Observed numeric metric value"
    )
    expected_value: Optional[float] = Field(
        None, description="Expected numeric metric value"
    )
    severity: Optional[Severity] = Field(
        None, description="Severity indicated by this evidence"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in this evidence [0,1]"
    )
    explanation: Optional[str] = Field(
        None, description="Explanation of why this record supports the diagnosis"
    )
    signal_name: Optional[str] = Field(None, description="Name of a supporting signal")
    signal_value: Optional[float] = Field(
        None, description="Numeric value of the supporting signal"
    )
    rule_id: Optional[str] = Field(None, description="Rule that produced this evidence")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional evidence attributes"
    )

    model_config = ConfigDict(from_attributes=True)


class RuleEvaluation(BaseModel):
    """Result of evaluating a deterministic rule against a session."""

    rule_id: str = Field(..., description="Stable rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    triggered: bool = Field(..., description="Whether the rule condition was met")
    score: Optional[float] = Field(None, description="Optional rule score")
    reason: Optional[str] = Field(None, description="Explanation of the evaluation result")

    model_config = ConfigDict(from_attributes=True)


class ExtractedSignal(BaseModel):
    """A normalized signal extracted from a session source record."""

    signal_name: str = Field(..., description="Stable signal identifier")
    value: float = Field(..., description="Numeric signal value")
    unit: Optional[str] = Field(None, description="Unit associated with the value")
    source: str = Field(..., description="Source that produced the signal")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in the extracted signal [0,1]"
    )

    model_config = ConfigDict(from_attributes=True)


class SimilarSession(BaseModel):
    """A historical session retrieved as similar to the current session."""

    session_id: UUID = Field(..., description="Identifier of the similar session")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score [0,1]"
    )
    failure_type: Optional[FailureType] = Field(
        None, description="Known failure type of the similar session"
    )
    summary: Optional[str] = Field(None, description="Safe summary of the similar session")

    model_config = ConfigDict(from_attributes=True)


FailureDiagnosis.model_rebuild()


class GroundTruth(BaseModel):
    """Expected outcome and injected scenario metadata for synthetic sessions."""

    expected_status: Optional[SessionStatus] = Field(
        None, description="Expected final session status"
    )
    expected_failure_type: Optional[FailureType] = Field(
        None, description="Expected root cause for evaluation"
    )
    expected_affected_component: Optional[ComponentType] = Field(
        None, description="Expected affected component for evaluation"
    )
    expected_severity: Optional[Severity] = Field(
        None, description="Expected severity for evaluation"
    )
    scenario_id: Optional[str] = Field(
        None, description="Identifier of the synthetic failure scenario"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Synthetic generation and evaluation metadata"
    )

    model_config = ConfigDict(from_attributes=True)


class InterviewSession(BaseModel):
    """Top-level representation of a synthetic interview session.

    This model aggregates transcript turns, LLM/tool calls, timeline events,
    computed metrics and an optional diagnosis. It is designed to be easy to
    serialize to JSON/JSONL for dataset interchange.
    """

    session_id: UUID = Field(..., description="UUID for the session")
    schema_version: Optional[str] = Field(
        None, description="Version of the schema used to serialize this session"
    )
    dataset_version: Optional[str] = Field(
        None, description="Version of the synthetic dataset containing this session"
    )
    pipeline_version: Optional[str] = Field(
        None, description="Version of the pipeline that produced derived outputs"
    )
    status: SessionStatus = Field(..., description="High-level session status")
    created_at: Optional[datetime] = Field(None, description="Session creation time")
    candidate: Optional[Candidate] = Field(None, description="Candidate metadata")
    transcript: List[TranscriptTurn] = Field(default_factory=list, description="Ordered transcript turns")
    llm_calls: List[LLMCall] = Field(default_factory=list, description="Recorded LLM calls")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Recorded tool calls")
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Reconstructed timeline events")
    metrics: Optional[SessionMetrics] = Field(None, description="Session aggregated metrics")
    extracted_signals: List[ExtractedSignal] = Field(
        default_factory=list,
        description="Normalized signals extracted from session records",
    )
    diagnosis: Optional[FailureDiagnosis] = Field(None, description="Optional failure diagnosis produced by the pipeline")
    ground_truth: Optional[GroundTruth] = Field(
        None, description="Expected labels and scenario data for synthetic evaluation"
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Free-form session metadata")

    model_config = ConfigDict(from_attributes=True, use_enum_values=False)
