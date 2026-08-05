"""Tests for standalone offline supervised model evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.dataset_loader import LabelSet, MLDataset
from ml.evaluate_models import ModelEvaluationError, ModelEvaluator


def _dataset() -> MLDataset:
    """Create a small balanced, deterministic multiclass feature dataset."""
    feature_names = tuple(f"signal_{index}" for index in range(31))
    rows = []
    categories = []
    for class_index, category in enumerate(("success", "timeout", "tool_failure")):
        for sample_index in range(4):
            row = np.full(31, float(class_index * 10 + sample_index))
            row[class_index] = float(class_index * 100 + sample_index)
            rows.append(row)
            categories.append(category)
    features = np.asarray(rows, dtype=float)
    features[0, 5] = np.nan
    return MLDataset(
        features=features,
        feature_names=feature_names,
        labels=LabelSet(
            root_causes=tuple(categories),
            affected_components=tuple("pipeline" for _ in categories),
            failure_categories=tuple(categories),
        ),
        session_ids=tuple(f"session-{index}" for index in range(len(categories))),
        source_format="test",
        issues=(),
    )


def test_successful_training_returns_all_supported_models() -> None:
    """All offline classifiers train and return structured evaluations."""
    report = ModelEvaluator(random_state=7).evaluate(_dataset())

    assert report.ranking == tuple(item.model_name for item in report.evaluations)
    assert set(report.ranking) == {
        "Logistic Regression", "Random Forest", "Gradient Boosting"
    }
    assert report.best_model.model_name == report.ranking[0]


def test_evaluation_is_deterministic() -> None:
    """Identical data and seeds produce identical evaluation reports."""
    evaluator = ModelEvaluator(random_state=11)

    assert evaluator.evaluate(_dataset()) == evaluator.evaluate(_dataset())


def test_confusion_matrices_match_class_dimension() -> None:
    """Every model reports a square matrix using the complete class set."""
    report = ModelEvaluator().evaluate(_dataset())

    for evaluation in report.evaluations:
        assert len(evaluation.class_labels) == 3
        assert len(evaluation.confusion_matrix) == 3
        assert all(len(row) == 3 for row in evaluation.confusion_matrix)


def test_model_ranking_and_metrics_are_complete() -> None:
    """Ranking follows macro F1 and every requested metric is present."""
    report = ModelEvaluator().evaluate(_dataset())
    macro_scores = [item.metrics["macro_f1"] for item in report.evaluations]

    assert macro_scores == sorted(macro_scores, reverse=True)
    required_metrics = {
        "accuracy", "precision", "recall", "f1_score", "macro_f1",
        "weighted_f1", "micro_f1",
    }
    assert required_metrics <= set(report.evaluations[0].metrics)


def test_empty_dataset_is_rejected() -> None:
    """Evaluation rejects an empty feature matrix before model fitting."""
    empty = MLDataset(
        features=np.empty((0, 31)), feature_names=tuple(f"signal_{index}" for index in range(31)),
        labels=LabelSet((), (), ()), session_ids=(), source_format="test", issues=(),
    )

    with pytest.raises(ModelEvaluationError):
        ModelEvaluator().evaluate(empty)


def test_invalid_labels_are_rejected() -> None:
    """Missing selected labels cannot be used for supervised evaluation."""
    dataset = _dataset()
    invalid = MLDataset(
        features=dataset.features, feature_names=dataset.feature_names,
        labels=LabelSet(dataset.labels.root_causes, dataset.labels.affected_components, (None,) * 12),
        session_ids=dataset.session_ids, source_format=dataset.source_format, issues=(),
    )

    with pytest.raises(ModelEvaluationError):
        ModelEvaluator().evaluate(invalid)


def test_evaluation_and_best_metadata_are_json_serializable(tmp_path: Path) -> None:
    """Reports serialize fully while persisted output contains best metadata only."""
    path = tmp_path / "best_model.json"
    report = ModelEvaluator().evaluate(_dataset(), metadata_path=path)

    assert json.loads(json.dumps(report.to_dict()))["ranking"]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == report.best_model.to_dict()
