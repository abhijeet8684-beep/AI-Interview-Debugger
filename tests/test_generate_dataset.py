"""Tests for deterministic synthetic interview dataset generation."""
from __future__ import annotations

from pathlib import Path

from generate_dataset import _generate_sessions, generate_synthetic_dataset
from pipeline.schema import InterviewSession


def test_generation_is_deterministic() -> None:
    """The same seed produces equivalent session records in the same order."""
    first_run = _generate_sessions(300, seed=101)
    second_run = _generate_sessions(300, seed=101)

    assert [session.model_dump_json() for session in first_run] == [
        session.model_dump_json() for session in second_run
    ]


def test_generation_produces_requested_session_count() -> None:
    """The generator produces exactly the supported requested session count."""
    sessions = _generate_sessions(300, seed=202)

    assert len(sessions) == 300


def test_generated_sessions_pass_schema_validation() -> None:
    """Every final session can be validated from its nested serialized form."""
    sessions = _generate_sessions(300, seed=303)

    for session in sessions:
        validated_session = InterviewSession.model_validate(session.model_dump())
        assert validated_session.diagnosis is None
        assert validated_session.ground_truth is not None


def test_generation_creates_jsonl_and_csv_files(tmp_path: Path) -> None:
    """Dataset generation writes both required non-empty output files."""
    jsonl_path, csv_path = generate_synthetic_dataset(tmp_path, n=300, seed=404)

    assert jsonl_path.exists()
    assert csv_path.exists()
    assert jsonl_path.read_text(encoding="utf-8").count("\n") == 300
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("session_id,")


def test_generation_creates_valid_single_session_examples(tmp_path: Path) -> None:
    """Representative examples are valid sessions and do not alter dataset files."""
    jsonl_path, csv_path = generate_synthetic_dataset(tmp_path, n=300, seed=505)
    examples_directory = tmp_path / "examples"

    assert examples_directory.is_dir()
    assert jsonl_path.read_text(encoding="utf-8").count("\n") == 300
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("session_id,")
    for filename in ("healthy_session.json", "llm_timeout.json", "tool_timeout.json"):
        session = InterviewSession.model_validate_json(
            (examples_directory / filename).read_text(encoding="utf-8")
        )
        assert session.session_id

    mixed_failure_path = examples_directory / "mixed_failure.json"
    if mixed_failure_path.exists():
        mixed_session = InterviewSession.model_validate_json(
            mixed_failure_path.read_text(encoding="utf-8")
        )
        assert mixed_session.session_id
