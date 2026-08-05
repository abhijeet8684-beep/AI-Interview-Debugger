"""Tests for standalone deterministic ML dataset drift detection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.dataset_loader import LabelSet, MLDataset
from ml.drift_detection import DriftDetectionError, DriftDetector


def _dataset(features: np.ndarray, categories: tuple[str, ...]) -> MLDataset:
    """Build a minimal labeled stable-feature dataset for drift tests."""
    count = len(categories)
    return MLDataset(
        features=features,
        feature_names=("latency", "tokens"),
        labels=LabelSet(
            root_causes=tuple("timeout" if item == "timeout" else None for item in categories),
            affected_components=tuple("pipeline" for _ in categories),
            failure_categories=categories,
        ),
        session_ids=tuple(f"session-{index}" for index in range(count)),
        source_format="test",
        issues=(),
    )


def test_identical_datasets_have_no_feature_drift() -> None:
    """Equal stable distributions return zero drift and no shifted features."""
    dataset = _dataset(np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]), ("success", "success", "timeout", "timeout"))

    report = DriftDetector().compare(dataset, dataset)

    assert report.summary.shifted_feature_count == 0
    assert report.summary.overall_drift_score == pytest.approx(0.0)
    assert all(not item.shifted for item in report.feature_drift)


def test_shifted_dataset_reports_feature_and_label_drift() -> None:
    """Large signal and class-distribution changes are independently reported."""
    baseline = _dataset(np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]), ("success", "success", "timeout", "timeout"))
    current = _dataset(np.asarray([[100.0, 10.0], [110.0, 20.0], [120.0, 30.0], [130.0, 40.0]]), ("timeout", "timeout", "timeout", "timeout"))

    report = DriftDetector().compare(baseline, current)

    assert report.feature_drift[0].shifted
    assert report.summary.largest_drift > 0.0
    category_change = next(item for item in report.label_drift if item.label_group == "failure_category" and item.label == "success")
    assert category_change.percentage_change == pytest.approx(-50.0)


def test_drift_report_is_json_serializable_and_exportable(tmp_path: Path) -> None:
    """In-memory and persisted reports use the same JSON-native representation."""
    dataset = _dataset(np.asarray([[1.0, 2.0], [3.0, 4.0]]), ("success", "timeout"))
    detector = DriftDetector()
    report = detector.compare(dataset, dataset)
    path = tmp_path / "drift.json"
    detector.export_json(report, path)

    assert json.loads(report.to_json()) == report.to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == report.to_dict()


def test_empty_datasets_are_handled_gracefully() -> None:
    """Matching empty datasets return a zero-valued, schema-consistent report."""
    empty = _dataset(np.empty((0, 2)), ())

    report = DriftDetector().compare(empty, empty)

    assert len(report.feature_drift) == 2
    assert report.summary.overall_drift_score == 0.0


def test_mismatched_feature_metadata_is_rejected() -> None:
    """Drift comparison requires the same stable feature ordering."""
    baseline = _dataset(np.asarray([[1.0, 2.0], [3.0, 4.0]]), ("success", "timeout"))
    invalid = MLDataset(
        features=baseline.features,
        feature_names=("different", "tokens"),
        labels=baseline.labels,
        session_ids=baseline.session_ids,
        source_format="test",
        issues=(),
    )

    with pytest.raises(DriftDetectionError):
        DriftDetector().compare(baseline, invalid)
