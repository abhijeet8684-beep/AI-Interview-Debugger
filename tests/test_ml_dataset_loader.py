"""Tests for standalone ML dataset loading and stable signal features."""
from __future__ import annotations

import csv
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from ml.dataset_loader import DatasetLoader, DatasetValidationError
from pipeline.schema import (
    ComponentType,
    ExtractedSignal,
    FailureType,
    GroundTruth,
    InterviewSession,
    SessionStatus,
)


def _session(index: int, with_signals: bool = False) -> InterviewSession:
    """Create one schema-valid ground-truth session for loader tests."""
    ground_truth = GroundTruth(
        expected_status=SessionStatus.FAILED,
        expected_failure_type=FailureType.TIMEOUT,
        expected_affected_component=ComponentType.INFRASTRUCTURE,
        scenario_id="llm_timeout",
    )
    return InterviewSession(
        session_id=uuid4(),
        status=SessionStatus.FAILED,
        ground_truth=ground_truth,
        extracted_signals=[
            ExtractedSignal(
                signal_name="total_tokens", value=float(100 + index),
                unit="tokens", source="test",
            )
        ] if with_signals else [],
    )


def test_jsonl_loading_preserves_order_labels_and_stable_features(tmp_path: Path) -> None:
    """JSONL rows remain ordered and use the complete frozen signal contract."""
    sessions = [_session(1), _session(2)]
    path = tmp_path / "sessions.jsonl"
    path.write_text("\n".join(item.model_dump_json() for item in sessions), encoding="utf-8")

    dataset = DatasetLoader().load(path)

    assert dataset.session_ids == tuple(str(item.session_id) for item in sessions)
    assert dataset.features.shape == (2, 31)
    assert dataset.feature_names == DatasetLoader().feature_names
    assert dataset.labels.root_causes == ("timeout", "timeout")
    assert dataset.labels.affected_components == ("infrastructure", "infrastructure")
    assert dataset.labels.failure_categories == ("llm_timeout", "llm_timeout")


def test_stored_signal_values_are_aligned_to_feature_metadata() -> None:
    """Existing extracted signals are reused without duplicate feature logic."""
    loader = DatasetLoader()
    dataset = loader.build_from_sessions([_session(5, with_signals=True)])
    token_index = dataset.feature_names.index("total_tokens")

    assert dataset.features[0, token_index] == pytest.approx(105.0)
    assert np.isnan(dataset.features[0, 0])


def test_malformed_jsonl_rows_are_reported_or_raise_in_strict_mode(tmp_path: Path) -> None:
    """Malformed JSONL rows are skipped deterministically unless strict mode is set."""
    valid = _session(1).model_dump_json()
    path = tmp_path / "malformed.jsonl"
    path.write_text(f"{valid}\n{{not-json}}\n", encoding="utf-8")
    loader = DatasetLoader()

    dataset = loader.load(path)

    assert dataset.features.shape[0] == 1
    assert len(dataset.issues) == 1
    assert dataset.issues[0].row_number == 2
    with pytest.raises(DatasetValidationError):
        loader.load(path, strict=True)


def test_csv_loading_maps_only_existing_signal_columns(tmp_path: Path) -> None:
    """Flattened CSV maps compatible signals and marks omitted signals as missing."""
    path = tmp_path / "sessions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "session_id", "duration_seconds", "num_turns", "num_llm_calls",
                "num_tool_calls", "avg_llm_latency_ms", "total_tokens",
                "ground_truth_failure_type", "ground_truth_component",
                "ground_truth_scenario",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "session_id": "session-a", "duration_seconds": "120", "num_turns": "8",
            "num_llm_calls": "2", "num_tool_calls": "1", "avg_llm_latency_ms": "450",
            "total_tokens": "900", "ground_truth_failure_type": "timeout",
            "ground_truth_component": "infrastructure", "ground_truth_scenario": "llm_timeout",
        })

    dataset = DatasetLoader().load(path)
    duration_index = dataset.feature_names.index("interview_duration_seconds")
    missing_index = dataset.feature_names.index("failed_llm_calls")

    assert dataset.source_format == "csv"
    assert dataset.features[0, duration_index] == pytest.approx(120.0)
    assert np.isnan(dataset.features[0, missing_index])
    assert dataset.labels.root_causes == ("timeout",)


def test_dataset_statistics_reports_balance_and_missing_values() -> None:
    """Statistics expose stable dimensions, class balance, and NaN counts."""
    dataset = DatasetLoader().build_from_sessions([_session(1, True)])
    statistics = DatasetLoader().statistics(dataset)

    assert statistics.feature_count == 31
    assert statistics.row_count == 1
    assert statistics.class_balance == {"timeout": 1}
    assert statistics.missing_values["total_tokens"] == 0
    assert statistics.missing_values["total_llm_calls"] == 1
    assert "31 stable features" in statistics.summary


def test_csv_requires_session_identifier_column(tmp_path: Path) -> None:
    """Invalid CSV schemas fail with an explicit validation error."""
    path = tmp_path / "invalid.csv"
    path.write_text("duration_seconds\n120\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        DatasetLoader().load(path)
