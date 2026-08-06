"""Tests for the ML analytics dashboard helpers."""
from __future__ import annotations

import numpy as np

from app.ml_analytics import (
    build_evaluation_dataframe,
    create_download_payload,
    dataset_cache_key,
    feature_importance_dataframe,
    get_confusion_matrix,
    load_dataset_statistics,
    load_json_report,
    run_ml_analytics,
    DEFAULT_DATASET_PATH,
)


def test_load_json_report_invalid_returns_none() -> None:
    assert load_json_report(b"not a json") is None


def test_build_evaluation_dataframe_returns_sorted_metrics() -> None:
    report = {
        "evaluations": [
            {
                "model_name": "Random Forest",
                "metrics": {
                    "accuracy": 0.8,
                    "precision": 0.75,
                    "recall": 0.76,
                    "macro_f1": 0.77,
                    "weighted_f1": 0.78,
                    "micro_f1": 0.79,
                },
            },
            {
                "model_name": "Logistic Regression",
                "metrics": {
                    "accuracy": 0.82,
                    "precision": 0.78,
                    "recall": 0.79,
                    "macro_f1": 0.80,
                    "weighted_f1": 0.81,
                    "micro_f1": 0.82,
                },
            },
        ]
    }
    df = build_evaluation_dataframe([report])
    assert list(df["Model"]) == ["Logistic Regression", "Random Forest"]
    assert df.iloc[0]["Macro F1"] == 0.8


def test_get_confusion_matrix_for_model() -> None:
    report = {
        "evaluations": [
            {
                "model_name": "Gradient Boosting",
                "confusion_matrix": [[10, 2], [1, 7]],
                "class_labels": ["success", "failure"],
            }
        ]
    }
    matrix, labels = get_confusion_matrix(report, "Gradient Boosting")
    assert matrix.shape == (2, 2)
    assert labels == ["success", "failure"]
    assert (matrix == [[10, 2], [1, 7]]).all()


def test_feature_importance_dataframe_returns_top_features() -> None:
    report = {
        "model_name": "Logistic Regression",
        "ranking": [
            {"feature_name": "f1", "importance_score": 0.6, "normalized_score": 0.5, "rank": 1},
            {"feature_name": "f2", "importance_score": 0.4, "normalized_score": 0.3, "rank": 2},
        ],
    }
    df = feature_importance_dataframe(report, top_n=2)
    first_row = df.iloc[0]
    second_row = df.iloc[1]

    assert first_row["Feature Name"] == "f1"
    assert first_row["Rank"] == 1
    assert first_row["Raw Importance"] == 0.6
    assert first_row["Normalized Importance"] == 0.5
    assert bool(first_row["Top 5"])
    assert second_row["Feature Name"] == "f2"


def test_load_dataset_statistics_default_sample() -> None:
    stats, dataset = load_dataset_statistics()
    assert stats.row_count > 0
    assert stats.feature_count > 0
    assert len(dataset.session_ids) == stats.row_count
    assert len(dataset.feature_names) == stats.feature_count


def test_dataset_cache_key_changes_for_different_content() -> None:
    first_key = dataset_cache_key("sessions.jsonl", b"first content")
    second_key = dataset_cache_key("sessions.jsonl", b"second content")
    assert first_key != second_key
    assert first_key == dataset_cache_key("sessions.jsonl", b"first content")


def test_create_download_payload_serializes_report() -> None:
    payload = create_download_payload({"model": "Logistic Regression", "accuracy": 0.92})
    assert isinstance(payload, str)
    assert '"model": "Logistic Regression"' in payload


def test_run_ml_analytics_generates_reports_for_sample_dataset() -> None:
    _, dataset = load_dataset_statistics(DEFAULT_DATASET_PATH)
    _, baseline_dataset = load_dataset_statistics(DEFAULT_DATASET_PATH)
    results = run_ml_analytics(dataset, baseline_dataset, random_state=42)

    assert "evaluation_report" in results
    assert "feature_importance_reports" in results
    assert "drift_report" in results
    assert "anomaly_report" in results
    assert isinstance(results["evaluation_report"], dict)
    assert isinstance(results["feature_importance_reports"], list)
    assert len(results["feature_importance_reports"]) >= 1
    assert results["evaluation_report"].get("evaluations")
