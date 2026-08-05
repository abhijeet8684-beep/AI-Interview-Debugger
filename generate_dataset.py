"""Generate deterministic synthetic sessions for the AI Interview Debugger.

The generator creates realistic, traceable interview records that exercise the
frozen pipeline schema. It intentionally writes ground truth only; predicted
diagnoses, rules, and derived signals are left to later pipeline stages.
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from faker import Faker

from pipeline.schema import (
    CallStatus,
    Candidate,
    ComponentType,
    ErrorType,
    EventType,
    FailureType,
    GroundTruth,
    InterviewSession,
    InterviewStage,
    LLMCall,
    SessionMetrics,
    SessionStatus,
    Severity,
    Speaker,
    TimelineEvent,
    ToolCall,
    TranscriptTurn,
)

DEFAULT_SESSION_COUNT = 400
MIN_SESSION_COUNT = 300
MAX_SESSION_COUNT = 500
DEFAULT_SEED = 42
SCHEMA_VERSION = "1.0"
DATASET_VERSION = "synthetic-v1"
PIPELINE_VERSION = "unprocessed"
DEFAULT_OUTPUT_DIRECTORY = Path("data") / "synthetic"


@dataclass(frozen=True)
class FailureScenario:
    """Configuration used to inject one realistic failure into a session."""

    scenario_id: str
    failure_type: Optional[FailureType]
    component: Optional[ComponentType]
    severity: Optional[Severity]
    error_type: Optional[ErrorType]
    error_message: Optional[str]
    target: str


SUCCESS_SCENARIO = FailureScenario(
    scenario_id="successful_interview",
    failure_type=None,
    component=None,
    severity=None,
    error_type=None,
    error_message=None,
    target="none",
)

FAILURE_SCENARIOS: tuple[FailureScenario, ...] = (
    FailureScenario(
        "speech_to_text_failure", FailureType.UNKNOWN, ComponentType.DATA,
        Severity.MEDIUM, ErrorType.EXECUTION,
        "Speech-to-text confidence dropped below the accepted threshold.", "transcript",
    ),
    FailureScenario(
        "llm_timeout", FailureType.TIMEOUT, ComponentType.MODEL_RESPONSE,
        Severity.HIGH, ErrorType.TIMEOUT,
        "The language model request exceeded the 30 second timeout.", "llm",
    ),
    FailureScenario(
        "tool_timeout", FailureType.TOOL_FAILURE, ComponentType.PIPELINE,
        Severity.HIGH, ErrorType.TIMEOUT,
        "The retrieval tool did not respond before its timeout.", "tool",
    ),
    FailureScenario(
        "invalid_json_response", FailureType.PROMPT_ISSUE,
        ComponentType.MODEL_RESPONSE, Severity.MEDIUM, ErrorType.VALIDATION,
        "The model response could not be parsed as the required JSON payload.", "llm",
    ),
    FailureScenario(
        "context_window_overflow", FailureType.TIMEOUT, ComponentType.PIPELINE,
        Severity.HIGH, ErrorType.VALIDATION,
        "The prompt exceeded the configured context window.", "llm",
    ),
    FailureScenario(
        "retrieval_failure", FailureType.TOOL_FAILURE, ComponentType.DATA,
        Severity.MEDIUM, ErrorType.EXECUTION,
        "No relevant knowledge-base documents were retrieved.", "tool",
    ),
    FailureScenario(
        "evaluation_failure", FailureType.UNKNOWN, ComponentType.PIPELINE,
        Severity.MEDIUM, ErrorType.VALIDATION,
        "The evaluation service returned an invalid score payload.", "tool",
    ),
    FailureScenario(
        "network_api_failure", FailureType.TIMEOUT, ComponentType.INFRASTRUCTURE,
        Severity.HIGH, ErrorType.PROVIDER,
        "The upstream API connection was reset by the peer.", "llm",
    ),
    FailureScenario(
        "database_failure", FailureType.TOOL_FAILURE, ComponentType.INFRASTRUCTURE,
        Severity.CRITICAL, ErrorType.EXECUTION,
        "The candidate-profile database query failed during evaluation.", "tool",
    ),
)

ROLES: tuple[str, ...] = (
    "Machine Learning Intern", "Data Science Intern", "Backend Engineering Intern",
    "Data Analyst Intern", "Software Engineering Intern",
)
SKILLS: tuple[str, ...] = (
    "Python", "SQL", "Pandas", "scikit-learn", "Statistics", "Docker", "FastAPI",
    "PyTorch", "Data Visualization", "System Design",
)
EDUCATION_LEVELS: tuple[str, ...] = (
    "B.Tech in Computer Science", "B.E. in Information Technology",
    "M.Sc. in Data Science", "B.Sc. in Statistics",
)
UNIVERSITIES: tuple[str, ...] = (
    "Indian Institute of Technology Delhi", "National Institute of Technology Karnataka",
    "Vellore Institute of Technology", "University of Delhi", "Pune University",
)
CERTIFICATIONS: tuple[str, ...] = (
    "Google Data Analytics", "AWS Certified Cloud Practitioner",
    "Microsoft Azure Fundamentals", "DeepLearning.AI Machine Learning Specialization",
)
COMPANIES: tuple[str, ...] = (
    "DataNest Analytics", "CloudScale Labs", "InsightWorks", "Vertex Systems",
    "Nimbus Retail", "Quantive Solutions",
)
RESPONSE_STYLES: tuple[str, ...] = (
    "excellent", "average", "weak", "incomplete", "verbose",
)

QUESTION_BANK: dict[InterviewStage, tuple[str, ...]] = {
    InterviewStage.INTRODUCTION: (
        "Please introduce yourself and describe a recent project you are proud of.",
        "What interests you about this {role} position?",
    ),
    InterviewStage.TECHNICAL: (
        "How would you evaluate a binary classifier with imbalanced classes?",
        "Explain the bias-variance trade-off using a practical example.",
        "How would you investigate a sudden drop in model precision?",
    ),
    InterviewStage.CODING: (
        "Write an approach to find duplicate records in a large dataset.",
        "How would you design a function to validate a JSON API response?",
        "Describe the time and space complexity of your proposed solution.",
    ),
    InterviewStage.BEHAVIORAL: (
        "Tell me about a time you received difficult feedback and acted on it.",
        "Describe how you handle disagreement with a teammate.",
    ),
    InterviewStage.EVALUATION: (
        "Is there anything you would improve about one of your earlier answers?",
    ),
}


def generate_synthetic_dataset(
    output_dir: Optional[Path] = None,
    n: int = DEFAULT_SESSION_COUNT,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """Generate and save deterministic synthetic interview sessions.

    Args:
        output_dir: Directory for output files. Defaults to ``data/synthetic``.
        n: Number of sessions to generate, constrained to 300 through 500.
        seed: Random seed used for all generated attributes.

    Returns:
        Paths to the JSONL dataset and flattened CSV summary, respectively.

    Raises:
        ValueError: If ``n`` is outside the supported dataset size range.
    """
    if not MIN_SESSION_COUNT <= n <= MAX_SESSION_COUNT:
        raise ValueError(
            f"n must be between {MIN_SESSION_COUNT} and {MAX_SESSION_COUNT}; got {n}."
        )

    destination = output_dir or DEFAULT_OUTPUT_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    sessions = _generate_sessions(n, seed)
    _validate_sessions(sessions)

    jsonl_path = destination / "interview_sessions.jsonl"
    csv_path = destination / "interview_sessions.csv"
    _write_jsonl(sessions, jsonl_path)
    _write_csv(sessions, csv_path)
    _write_example_sessions(sessions, _examples_directory(destination))
    _print_statistics(_calculate_statistics(sessions))
    return jsonl_path, csv_path


def _generate_sessions(n: int, seed: int) -> list[InterviewSession]:
    """Generate session models from isolated deterministic random sources."""
    randomizer = random.Random(seed)
    faker = Faker("en_IN")
    faker.seed_instance(seed)
    return [_build_session(index, randomizer, faker, seed) for index in range(n)]


def _build_session(
    index: int,
    randomizer: random.Random,
    faker: Faker,
    seed: int,
) -> InterviewSession:
    """Build one complete synthetic interview session from a deterministic stream."""
    scenario = _choose_scenario(index, randomizer)
    session_id = uuid5(NAMESPACE_URL, f"ai-interview-debugger/{seed}/{index}")
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        minutes=index * 13 + randomizer.randint(0, 11)
    )
    candidate = _build_candidate(index, randomizer, faker)
    role = randomizer.choice(ROLES)
    stages = _select_stages(randomizer)
    trace_id = f"trace_{session_id.hex[:12]}"
    failure_stage_index = randomizer.randrange(1, len(stages)) if scenario != SUCCESS_SCENARIO else -1
    cascade_failure = scenario != SUCCESS_SCENARIO and randomizer.random() < 0.45

    transcript, llm_calls, tool_calls, timeline = _build_interview_records(
        candidate, role, stages, scenario, failure_stage_index, trace_id, start_time,
        cascade_failure, randomizer,
    )
    is_failed = scenario != SUCCESS_SCENARIO
    status = SessionStatus.FAILED if is_failed else SessionStatus.PASSED
    metrics = _build_metrics(transcript, llm_calls, tool_calls, is_failed)
    ground_truth = GroundTruth(
        expected_status=status,
        expected_failure_type=scenario.failure_type,
        expected_affected_component=scenario.component,
        expected_severity=scenario.severity,
        scenario_id=scenario.scenario_id,
        metadata={
            "generator_seed": seed,
            "failure_target": scenario.target,
            "retry_count": _count_retries(timeline),
            "cascade_failure": cascade_failure,
        },
    )
    return InterviewSession(
        session_id=session_id,
        schema_version=SCHEMA_VERSION,
        dataset_version=DATASET_VERSION,
        pipeline_version=PIPELINE_VERSION,
        status=status,
        created_at=start_time,
        candidate=candidate,
        transcript=transcript,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        timeline=timeline,
        metrics=metrics,
        extracted_signals=[],
        diagnosis=None,
        ground_truth=ground_truth,
        metadata={
            "interview_role": role,
            "trace_id": trace_id,
            "stages": [stage.value for stage in stages],
            "generator": "synthetic_interview_generator",
        },
    )


def _choose_scenario(index: int, randomizer: random.Random) -> FailureScenario:
    """Return a balanced scenario selection with deterministic random ordering."""
    scenarios: Sequence[FailureScenario] = (SUCCESS_SCENARIO,) + FAILURE_SCENARIOS
    base_scenario = scenarios[index % len(scenarios)]
    return base_scenario if randomizer.random() < 0.7 else randomizer.choice(scenarios)


def _build_candidate(index: int, randomizer: random.Random, faker: Faker) -> Candidate:
    """Create a varied candidate profile without using personally identifiable data."""
    skills = randomizer.sample(SKILLS, k=randomizer.randint(3, 5))
    return Candidate(
        candidate_id=f"cand_{index + 1:04d}",
        name=faker.name(),
        experience_years=round(randomizer.uniform(0.0, 4.5), 1),
        metadata={
            "skills": skills,
            "location": randomizer.choice(("Remote", "Bengaluru", "Pune", "Delhi")),
            "education": randomizer.choice(EDUCATION_LEVELS),
            "university": randomizer.choice(UNIVERSITIES),
            "certifications": randomizer.sample(CERTIFICATIONS, k=randomizer.randint(0, 2)),
            "previous_company": randomizer.choice(COMPANIES),
            "project": {
                "name": faker.catch_phrase(),
                "domain": randomizer.choice(("analytics", "machine_learning", "backend", "data_engineering")),
                "impact": f"Improved a synthetic KPI by {randomizer.randint(8, 35)}%.",
            },
        },
    )


def _select_stages(randomizer: random.Random) -> list[InterviewStage]:
    """Select an ordered, multi-stage interview structure."""
    middle_stages = [InterviewStage.TECHNICAL, InterviewStage.CODING, InterviewStage.BEHAVIORAL]
    randomizer.shuffle(middle_stages)
    return [InterviewStage.INTRODUCTION, *middle_stages[:randomizer.randint(2, 3)], InterviewStage.EVALUATION]


def _build_interview_records(
    candidate: Candidate,
    role: str,
    stages: Sequence[InterviewStage],
    scenario: FailureScenario,
    failure_stage_index: int,
    trace_id: str,
    start_time: datetime,
    cascade_failure: bool,
    randomizer: random.Random,
) -> tuple[list[TranscriptTurn], list[LLMCall], list[ToolCall], list[TimelineEvent]]:
    """Create transcript, call records, and chronologically ordered events."""
    transcript: list[TranscriptTurn] = []
    llm_calls: list[LLMCall] = []
    tool_calls: list[ToolCall] = []
    timeline: list[TimelineEvent] = []
    current_time = start_time
    event_number = 0

    def add_event(
        event_type: EventType,
        description: str,
        stage: InterviewStage,
        related_turn_id: Optional[int] = None,
        related_call_id: Optional[str] = None,
        advance_seconds: int = 4,
    ) -> None:
        nonlocal current_time, event_number
        event_number += 1
        timeline.append(TimelineEvent(
            event_id=f"event_{event_number}", trace_id=trace_id, stage=stage,
            timestamp=current_time, event_type=event_type, description=description,
            related_turn_id=related_turn_id, related_call_id=related_call_id,
        ))
        current_time += timedelta(seconds=advance_seconds)

    add_event(EventType.TURN, "Interview started.", InterviewStage.INTRODUCTION)
    turn_id = 0
    failed = False
    failure_recorded = False
    for stage_index, stage in enumerate(stages):
        question_count = 1 if stage in (InterviewStage.INTRODUCTION, InterviewStage.EVALUATION) else randomizer.randint(1, 2)
        for question_index in range(question_count):
            turn_id += 1
            question = randomizer.choice(QUESTION_BANK[stage]).format(role=role)
            interviewer_turn = TranscriptTurn(
                turn_id=turn_id, trace_id=trace_id, stage=stage,
                speaker=Speaker.INTERVIEWER, text=question, timestamp=current_time,
                tokens=_estimate_tokens(question), annotations={"question_index": question_index},
            )
            transcript.append(interviewer_turn)
            add_event(EventType.TURN, "Interviewer asked a question.", stage, turn_id)

            response_style = randomizer.choice(RESPONSE_STYLES)
            candidate_response = _candidate_response(candidate, stage, response_style, randomizer)
            if scenario.target == "transcript" and stage_index == failure_stage_index:
                candidate_response = "[inaudible] ... model training ... [uncertain transcription]"
            turn_id += 1
            candidate_turn = TranscriptTurn(
                turn_id=turn_id, trace_id=trace_id, stage=stage,
                speaker=Speaker.CANDIDATE, text=candidate_response, timestamp=current_time,
                tokens=_estimate_tokens(candidate_response),
                annotations={
                    "response_style": response_style,
                    "speech_confidence": 0.34 if scenario.target == "transcript" and stage_index == failure_stage_index else round(randomizer.uniform(0.86, 0.99), 2),
                },
            )
            transcript.append(candidate_turn)
            add_event(EventType.TURN, "Candidate response received.", stage, turn_id, advance_seconds=randomizer.randint(12, 35))

            llm_call = _build_llm_call(
                len(llm_calls) + 1, candidate_turn, stage, scenario,
                stage_index == failure_stage_index and scenario.target == "llm", trace_id,
                current_time, randomizer,
            )
            llm_calls.append(llm_call)
            add_event(EventType.LLM_CALL, "Interview response evaluated by LLM.", stage, candidate_turn.turn_id, llm_call.call_id)

            if stage in (InterviewStage.TECHNICAL, InterviewStage.CODING, InterviewStage.EVALUATION):
                tool_call = _build_tool_call(
                    len(tool_calls) + 1, candidate_turn, stage, scenario,
                    stage_index == failure_stage_index
                    and (scenario.target == "tool" or (cascade_failure and scenario.target == "llm")), trace_id,
                    current_time, llm_call.call_id, randomizer,
                )
                tool_calls.append(tool_call)
                add_event(EventType.TOOL_CALL, f"{tool_call.tool_name} completed.", stage, candidate_turn.turn_id, tool_call.call_id)

            if _is_failure_at_stage(scenario, stage_index, failure_stage_index) and not failure_recorded:
                failed = True
                failure_recorded = True
                add_event(EventType.ERROR, scenario.error_message or "Interview processing failed.", stage, candidate_turn.turn_id, advance_seconds=2)
                _append_retry_events(add_event, stage, candidate_turn.turn_id, scenario, randomizer)
                if cascade_failure:
                    add_event(
                        EventType.ERROR,
                        "Failure propagated to downstream interview processing.",
                        stage,
                        candidate_turn.turn_id,
                        advance_seconds=2,
                    )
                else:
                    break
        if failed and not cascade_failure:
            break

    final_stage = stages[min(failure_stage_index, len(stages) - 1)] if failed else InterviewStage.EVALUATION
    final_description = "Interview failed after an unrecoverable processing error." if failed else "Interview completed successfully."
    add_event(EventType.DIAGNOSIS, final_description, final_stage, advance_seconds=0)
    return transcript, llm_calls, tool_calls, timeline


def _build_llm_call(
    call_number: int,
    candidate_turn: TranscriptTurn,
    stage: InterviewStage,
    scenario: FailureScenario,
    inject_failure: bool,
    trace_id: str,
    timestamp: datetime,
    randomizer: random.Random,
) -> LLMCall:
    """Create an LLM call, optionally reflecting a scenario-specific failure."""
    latency = randomizer.uniform(350, 1800)
    status = CallStatus.SUCCESS
    response = '{"assessment": "response received", "score": 0.78}'
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    if inject_failure:
        latency = randomizer.uniform(30_000, 45_000) if scenario.error_type == ErrorType.TIMEOUT else randomizer.uniform(800, 2400)
        status = CallStatus.TIMEOUT if scenario.error_type == ErrorType.TIMEOUT else CallStatus.ERROR
        response = "{assessment: invalid}" if scenario.error_type == ErrorType.VALIDATION else None
        error_type = scenario.error_type
        error_message = scenario.error_message
    return LLMCall(
        call_id=f"llm_{call_number}", trace_id=trace_id, related_turn_id=candidate_turn.turn_id,
        model_name=randomizer.choice(("gpt-4o-mini", "gpt-4.1-mini", "interview-evaluator-v1")),
        prompt=f"Evaluate the candidate response for the {stage.value} stage.", response=response,
        timestamp=timestamp, latency_ms=round(latency, 2), duration_ms=round(latency, 2),
        status=status, error_type=error_type, error_message=error_message,
        tokens_input=randomizer.randint(180, 900), tokens_output=randomizer.randint(60, 250),
        metadata={"provider": "synthetic-provider", "stage": stage.value},
    )


def _build_tool_call(
    call_number: int,
    candidate_turn: TranscriptTurn,
    stage: InterviewStage,
    scenario: FailureScenario,
    inject_failure: bool,
    trace_id: str,
    timestamp: datetime,
    parent_call_id: str,
    randomizer: random.Random,
) -> ToolCall:
    """Create a tool call, optionally reflecting a scenario-specific failure."""
    tool_name = randomizer.choice(("knowledge_base_retrieval", "code_evaluator", "profile_database"))
    duration = randomizer.uniform(80, 900)
    status: CallStatus = CallStatus.SUCCESS
    output: Optional[dict[str, Any]] = {"result": "ok", "items_returned": randomizer.randint(1, 5)}
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    if inject_failure:
        duration = randomizer.uniform(15_000, 25_000) if scenario.error_type == ErrorType.TIMEOUT else randomizer.uniform(250, 1500)
        status = CallStatus.TIMEOUT if scenario.error_type == ErrorType.TIMEOUT else CallStatus.ERROR
        output = None
        error_type = scenario.error_type
        error_message = scenario.error_message
    return ToolCall(
        call_id=f"tool_{call_number}", trace_id=trace_id, related_turn_id=candidate_turn.turn_id,
        parent_call_id=parent_call_id, tool_name=tool_name,
        input={"stage": stage.value, "candidate_turn_id": candidate_turn.turn_id}, output=output,
        timestamp=timestamp, duration_ms=round(duration, 2), status=status,
        error_type=error_type, error_message=error_message,
        metadata={"retryable": inject_failure and scenario.error_type == ErrorType.TIMEOUT},
    )


def _append_retry_events(
    add_event: Callable[..., None],
    stage: InterviewStage,
    turn_id: int,
    scenario: FailureScenario,
    randomizer: random.Random,
) -> None:
    """Add timeline-only retry attempts after retryable operational failures."""
    if scenario.error_type not in (ErrorType.TIMEOUT, ErrorType.PROVIDER):
        return
    for retry_number in range(randomizer.randint(1, 2)):
        add_event(
            EventType.LLM_CALL if scenario.target == "llm" else EventType.TOOL_CALL,
            f"Retry attempt {retry_number + 1} started.", stage, turn_id, advance_seconds=3,
        )


def _is_failure_at_stage(
    scenario: FailureScenario, stage_index: int, failure_stage_index: int
) -> bool:
    """Return whether the configured scenario should terminate the session now."""
    return scenario != SUCCESS_SCENARIO and stage_index == failure_stage_index


def _candidate_response(
    candidate: Candidate,
    stage: InterviewStage,
    response_style: str,
    randomizer: random.Random,
) -> str:
    """Return a concise, plausible response tailored to the interview stage."""
    skill = randomizer.choice(candidate.metadata.get("skills", ["Python"])) if candidate.metadata else "Python"
    core_responses = {
        InterviewStage.INTRODUCTION: f"I have focused on {skill} projects and enjoy turning ambiguous data problems into measurable outcomes.",
        InterviewStage.TECHNICAL: "I would validate the data split, inspect precision and recall, then compare performance across meaningful segments.",
        InterviewStage.CODING: "I would use a hash-based pass, define edge cases first, and validate the result with representative tests.",
        InterviewStage.BEHAVIORAL: "I would clarify the disagreement, share evidence, and align on a small experiment before committing to a direction.",
        InterviewStage.EVALUATION: "I would make the assumptions explicit and communicate how I would validate them after deployment.",
    }
    core_response = core_responses[stage]
    style_prefixes = {
        "excellent": "I would approach it systematically. ",
        "average": "My initial approach would be to ",
        "weak": "I think I would probably ",
        "incomplete": "I would start by ",
        "verbose": "There are several considerations here, including scope, assumptions, validation, and communication. ",
    }
    response = f"{style_prefixes[response_style]}{core_response}"
    if response_style == "weak":
        return response.split(",")[0] + "."
    if response_style == "incomplete":
        return response.split(" then ")[0] + "."
    if response_style == "verbose":
        return f"{response} I would document the trade-offs, monitor the outcome, and iterate with stakeholders."
    return response


def _build_metrics(
    transcript: Sequence[TranscriptTurn],
    llm_calls: Sequence[LLMCall],
    tool_calls: Sequence[ToolCall],
    failed: bool,
) -> SessionMetrics:
    """Create aggregate metrics directly from generated source records."""
    token_counts = [turn.tokens or 0 for turn in transcript]
    latencies = [call.latency_ms for call in llm_calls if call.latency_ms is not None]
    errors = sum(call.status != CallStatus.SUCCESS for call in llm_calls) + sum(
        call.status != CallStatus.SUCCESS for call in tool_calls
    )
    return SessionMetrics(
        num_turns=len(transcript), avg_turn_length=round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0,
        avg_llm_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        total_tokens=sum(token_counts) + sum((call.tokens_input or 0) + (call.tokens_output or 0) for call in llm_calls),
        num_errors=errors + int(failed and errors == 0),
        metadata={"turn_length_unit": "tokens", "tool_call_count": len(tool_calls)},
    )


def _count_retries(timeline: Sequence[TimelineEvent]) -> int:
    """Count retry events for ground-truth generation metadata."""
    return sum("Retry attempt" in (event.description or "") for event in timeline)


def _estimate_tokens(text: str) -> int:
    """Estimate token count consistently without a model tokenizer dependency."""
    return max(1, round(len(text.split()) * 1.3))


def _validate_sessions(sessions: Sequence[InterviewSession]) -> None:
    """Re-validate every generated session before any output is written.

    Construction already validates individual models, but this pass validates
    the final nested serialization shape to catch accidental contract drift.

    Args:
        sessions: Fully generated session models.

    Raises:
        ValueError: If a session cannot be re-created from its serialized data.
    """
    for index, session in enumerate(sessions):
        try:
            InterviewSession.model_validate(session.model_dump())
        except Exception as error:
            raise ValueError(f"Generated session at index {index} is invalid.") from error


def _calculate_statistics(sessions: Sequence[InterviewSession]) -> dict[str, Any]:
    """Calculate concise aggregate statistics for the generated dataset."""
    scenario_counts: dict[str, int] = {}
    durations: list[float] = []
    latencies: list[float] = []
    for session in sessions:
        scenario = session.ground_truth.scenario_id if session.ground_truth else "unknown"
        if session.status == SessionStatus.FAILED:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        timestamps = [event.timestamp for event in session.timeline if event.timestamp is not None]
        if len(timestamps) > 1:
            durations.append((max(timestamps) - min(timestamps)).total_seconds())
        latencies.extend(
            call.latency_ms for call in session.llm_calls if call.latency_ms is not None
        )
    return {
        "total_sessions": len(sessions),
        "success_count": sum(session.status == SessionStatus.PASSED for session in sessions),
        "failure_counts": dict(sorted(scenario_counts.items())),
        "average_duration_seconds": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "average_llm_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }


def _print_statistics(statistics: dict[str, Any]) -> None:
    """Print the required dataset summary in a compact, stable format."""
    print("Synthetic dataset summary")
    print(f"  Total sessions: {statistics['total_sessions']}")
    print(f"  Successful sessions: {statistics['success_count']}")
    print(f"  Failure counts by scenario: {statistics['failure_counts']}")
    print(f"  Average interview duration: {statistics['average_duration_seconds']} seconds")
    print(f"  Average LLM latency: {statistics['average_llm_latency_ms']} ms")


def _write_jsonl(sessions: Sequence[InterviewSession], destination: Path) -> None:
    """Write complete session objects as newline-delimited JSON."""
    with destination.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(session.model_dump_json())
            handle.write("\n")


def _write_csv(sessions: Sequence[InterviewSession], destination: Path) -> None:
    """Write a flat session-level CSV for quick analysis and dashboard previews."""
    fieldnames = [
        "session_id", "status", "candidate_id", "candidate_name", "experience_years",
        "interview_role", "stages", "duration_seconds", "num_turns", "num_llm_calls",
        "num_tool_calls", "num_errors", "avg_llm_latency_ms", "total_tokens",
        "ground_truth_scenario", "ground_truth_failure_type", "ground_truth_component",
        "ground_truth_severity", "schema_version", "dataset_version", "pipeline_version",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for session in sessions:
            writer.writerow(_session_row(session))


def _examples_directory(dataset_directory: Path) -> Path:
    """Return the repository examples directory for default dataset generation.

    Custom output directories keep generated examples alongside their dataset to
    make test and ad-hoc generation runs self-contained.
    """
    if dataset_directory.resolve() == DEFAULT_OUTPUT_DIRECTORY.resolve():
        return dataset_directory.parents[1] / "examples"
    return dataset_directory / "examples"


def _write_example_sessions(
    sessions: Sequence[InterviewSession], destination: Path
) -> None:
    """Export representative generated sessions as single-object JSON files."""
    selectors = {
        "healthy_session.json": lambda session: session.status == SessionStatus.PASSED,
        "llm_timeout.json": lambda session: _scenario_id(session) == "llm_timeout",
        "tool_timeout.json": lambda session: _scenario_id(session) == "tool_timeout",
        "mixed_failure.json": _is_mixed_failure_session,
    }
    destination.mkdir(parents=True, exist_ok=True)
    print("Created examples:")
    for filename, selector in selectors.items():
        session = next((item for item in sessions if selector(item)), None)
        if session is None:
            print(f"Warning: no representative session available for {filename}.")
            continue
        (destination / filename).write_text(
            session.model_dump_json(), encoding="utf-8"
        )
        print(f"✓ {filename}")


def _scenario_id(session: InterviewSession) -> Optional[str]:
    """Return the generated scenario identifier when ground truth is available."""
    return session.ground_truth.scenario_id if session.ground_truth else None


def _is_mixed_failure_session(session: InterviewSession) -> bool:
    """Return whether a generated session contains multiple failed call records."""
    failed_calls = sum(
        call.status not in (CallStatus.SUCCESS, CallStatus.OK)
        for call in [*session.llm_calls, *session.tool_calls]
    )
    return failed_calls >= 2


def _session_row(session: InterviewSession) -> dict[str, Any]:
    """Flatten one session into a stable CSV row without duplicating raw logs."""
    candidate = session.candidate
    ground_truth = session.ground_truth
    timestamps = [event.timestamp for event in session.timeline if event.timestamp is not None]
    duration = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0.0
    return {
        "session_id": str(session.session_id), "status": session.status.value,
        "candidate_id": candidate.candidate_id if candidate else None,
        "candidate_name": candidate.name if candidate else None,
        "experience_years": candidate.experience_years if candidate else None,
        "interview_role": (session.metadata or {}).get("interview_role"),
        "stages": ";".join((session.metadata or {}).get("stages", [])),
        "duration_seconds": round(duration, 2), "num_turns": session.metrics.num_turns if session.metrics else None,
        "num_llm_calls": len(session.llm_calls), "num_tool_calls": len(session.tool_calls),
        "num_errors": session.metrics.num_errors if session.metrics else None,
        "avg_llm_latency_ms": session.metrics.avg_llm_latency_ms if session.metrics else None,
        "total_tokens": session.metrics.total_tokens if session.metrics else None,
        "ground_truth_scenario": ground_truth.scenario_id if ground_truth else None,
        "ground_truth_failure_type": ground_truth.expected_failure_type.value if ground_truth and ground_truth.expected_failure_type else None,
        "ground_truth_component": ground_truth.expected_affected_component.value if ground_truth and ground_truth.expected_affected_component else None,
        "ground_truth_severity": ground_truth.expected_severity.value if ground_truth and ground_truth.expected_severity else None,
        "schema_version": session.schema_version, "dataset_version": session.dataset_version,
        "pipeline_version": session.pipeline_version,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for deterministic generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic interview sessions")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIRECTORY, help="Output directory")
    parser.add_argument("--n", type=int, default=DEFAULT_SESSION_COUNT, help="Session count (300-500)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed")
    return parser.parse_args()


def main() -> None:
    """Run the synthetic dataset generator CLI."""
    args = _parse_args()
    jsonl_path, csv_path = generate_synthetic_dataset(args.out, args.n, args.seed)
    print(f"Wrote {args.n} sessions to {jsonl_path} and {csv_path}")


if __name__ == "__main__":
    main()
