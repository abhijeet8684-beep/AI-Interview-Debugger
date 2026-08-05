"""Tests for standalone deterministic ML anomaly and cluster analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.anomaly_detection import AnomalyDetectionError, AnomalyDetector
from ml.dataset_loader import LabelSet, MLDataset


def _dataset() -> MLDataset:
    """Build clustered data with one remote historical session."""
    first_cluster = np.asarray([[0.0, 0.1], [0.1, 0.0], [-0.1, 0.0], [0.0, -0.1], [0.05, 0.05]])
    second_cluster = np.asarray([[3.0, 3.1], [3.1, 3.0], [2.9, 3.0], [3.0, 2.9], [3.05, 3.05]])
    features = np.vstack([first_cluster, second_cluster, [[20.0, 20.0]]])
    count = len(features)
    return MLDataset(
        features=features,
        feature_names=("latency", "tokens"),
        labels=LabelSet((None,) * count, (None,) * count, (None,) * count),
        session_ids=tuple(f"session-{index}" for index in range(count)),
        source_format="test",
        issues=(),
    )


def test_isolation_forest_returns_scores_and_flags() -> None:
    """Each offline session receives a finite score and deterministic flag."""
    report = AnomalyDetector(contamination=0.1, min_samples=3).analyze(_dataset())

    assert len(report.records) == 11
    assert all(np.isfinite(item.anomaly_score) for item in report.records)
    assert report.anomaly_count >= 1


def test_dbscan_clustering_is_deterministic() -> None:
    """Cluster labels and unknown-failure flags repeat for fixed inputs."""
    detector = AnomalyDetector(contamination=0.1, min_samples=3)

    assert detector.analyze(_dataset()) == detector.analyze(_dataset())


def test_noise_points_and_cluster_statistics_are_exposed() -> None:
    """DBSCAN noise and per-cluster counts are preserved in the report."""
    report = AnomalyDetector(contamination=0.1, min_samples=3).analyze(_dataset())

    assert report.cluster_statistics
    assert any(item.is_noise for item in report.records)
    assert report.noise_point_count >= 1
    assert report.unknown_failure_count >= report.anomaly_count


def test_empty_dataset_returns_empty_report() -> None:
    """Empty historical data is handled without fitting sklearn estimators."""
    empty = MLDataset(
        features=np.empty((0, 2)), feature_names=("latency", "tokens"),
        labels=LabelSet((), (), ()), session_ids=(), source_format="test", issues=(),
    )

    report = AnomalyDetector().analyze(empty)

    assert report.records == ()
    assert report.cluster_statistics == ()


def test_anomaly_report_is_json_serializable_and_exportable(tmp_path: Path) -> None:
    """Persisted anomaly reports retain their complete JSON-native content."""
    detector = AnomalyDetector(contamination=0.1, min_samples=3)
    report = detector.analyze(_dataset())
    path = tmp_path / "anomalies.json"
    detector.export_json(report, path)

    assert json.loads(report.to_json()) == report.to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == report.to_dict()


def test_unsupported_feature_shape_is_rejected() -> None:
    """Anomaly analysis requires a two-dimensional stable feature matrix."""
    invalid = MLDataset(
        features=np.asarray([1.0, 2.0]), feature_names=("latency", "tokens"),
        labels=LabelSet((None,), (None,), (None,)), session_ids=("session",),
        source_format="test", issues=(),
    )

    with pytest.raises(AnomalyDetectionError):
        AnomalyDetector().analyze(invalid)
