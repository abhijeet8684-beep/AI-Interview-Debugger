"""Tests for deterministic offline sklearn feature-importance analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.feature_importance import FeatureImportanceAnalyzer, FeatureImportanceError


def _training_data() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Create a deterministic multiclass dataset for fitted estimator tests."""
    features, labels = make_classification(
        n_samples=90,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    return features, labels, tuple(f"signal_{index}" for index in range(8))


def test_random_forest_native_importance() -> None:
    """Random Forest native importances produce a complete normalized ranking."""
    features, labels, names = _training_data()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(features, labels)

    report = FeatureImportanceAnalyzer().analyze(model, names)

    assert report.metadata["method"] == "native"
    assert len(report.ranking) == len(names)
    assert sum(item.normalized_score for item in report.ranking) == pytest.approx(1.0)


def test_gradient_boosting_native_importance() -> None:
    """Gradient Boosting is detected through its native importance interface."""
    features, labels, names = _training_data()
    model = GradientBoostingClassifier(random_state=42).fit(features, labels)

    report = FeatureImportanceAnalyzer().analyze(model, names)

    assert report.model_name == "GradientBoostingClassifier"
    assert report.summary.top_features


def test_logistic_regression_coefficient_importance() -> None:
    """Linear coefficients are converted to absolute normalized importance."""
    features, labels, names = _training_data()
    model = LogisticRegression(max_iter=1_000, random_state=42).fit(features, labels)

    report = FeatureImportanceAnalyzer().analyze(model, names)

    assert report.metadata["method"] == "coefficient"
    assert all(item.importance_score >= 0.0 for item in report.ranking)
    assert sum(item.normalized_score for item in report.ranking) == pytest.approx(1.0)


def test_permutation_importance_is_deterministic() -> None:
    """Permutation importance uses a fixed random state and stable ranking."""
    features, labels, names = _training_data()
    model = LogisticRegression(max_iter=1_000, random_state=42).fit(features, labels)
    analyzer = FeatureImportanceAnalyzer(random_state=42, n_repeats=5)

    first = analyzer.analyze(model, names, features, labels, method="permutation")
    second = analyzer.analyze(model, names, features, labels, method="permutation")

    assert first == second
    assert first.metadata["method"] == "permutation"


def test_ranking_uses_stable_score_and_name_ordering() -> None:
    """Repeated native analysis preserves the exact ranked feature order."""
    features, labels, names = _training_data()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(features, labels)
    analyzer = FeatureImportanceAnalyzer()

    assert analyzer.analyze(model, names).ranking == analyzer.analyze(model, names).ranking


def test_report_json_serialization_and_export(tmp_path: Path) -> None:
    """In-memory and persisted explainability reports use JSON-native values."""
    features, labels, names = _training_data()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(features, labels)
    analyzer = FeatureImportanceAnalyzer()
    report = analyzer.analyze(model, names)
    path = tmp_path / "importance.json"
    analyzer.export_json(report, path)

    assert json.loads(report.to_json()) == report.to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == report.to_dict()


def test_unsupported_estimator_is_rejected() -> None:
    """Estimators without an importance interface fail with a clear error."""
    with pytest.raises(FeatureImportanceError):
        FeatureImportanceAnalyzer().analyze(object(), ("signal",))


def test_unfitted_model_is_rejected() -> None:
    """An unfitted estimator cannot provide feature importance."""
    with pytest.raises(FeatureImportanceError):
        FeatureImportanceAnalyzer().analyze(LogisticRegression(), ("signal",))
