"""Tests for pure ML Analytics dashboard data-preparation helpers."""
from __future__ import annotations

import numpy as np

from app.ml_analytics import (
    _largest_shifted_feature,
    build_analytics_summary,
    build_anomaly_summary,
    build_confusion_matrix_observations,
    build_dataset_card_values,
    build_feature_importance_reports,
    build_model_insights,
    feature_importance_dataframe,
    format_confusion_matrix_label,
    style_model_evaluation_dataframe,
    summarize_confusion_matrix,
)
from ml.dataset_loader import DatasetStatistics, LabelSet, MLDataset


def _dataset() -> MLDataset:
    """Create a compact dataset for dashboard helper tests."""
    return MLDataset(
        features=np.zeros((4, 2)),
        feature_names=("latency", "tokens"),
        labels=LabelSet((None,) * 4, (None,) * 4, ("success", "timeout", "timeout", "success")),
        session_ids=("a", "b", "c", "d"),
        source_format="test",
        issues=(),
    )


def test_confusion_summary_identifies_largest_dynamic_confusion() -> None:
    """The matrix summary names the largest off-diagonal label pair."""
    summary = summarize_confusion_matrix(
        np.asarray([[8, 3], [1, 6]]), ["success", "timeout"]
    )

    assert "success with timeout (3 sessions)" in summary


def test_confusion_summary_handles_perfect_matrix() -> None:
    """A diagonal-only matrix reports no observed class confusion."""
    assert "No class confusions" in summarize_confusion_matrix(
        np.asarray([[3, 0], [0, 4]]), ["success", "timeout"]
    )


def test_feature_importance_table_rounds_and_marks_top_five() -> None:
    """Feature rows retain rank and expose rounded raw/normalized scores."""
    report = {
        "model_name": "RandomForestClassifier",
        "ranking": [
            {"feature_name": "latency", "importance_score": 0.12345, "normalized_score": 0.67891, "rank": 1},
            {"feature_name": "tokens", "importance_score": 0.00234, "normalized_score": 0.00345, "rank": 6},
        ],
    }

    dataframe = feature_importance_dataframe(report)

    assert list(dataframe["Top 5"]) == [True, False]
    assert list(dataframe["Top 3"]) == [True, False]
    assert dataframe.iloc[0]["Raw Importance"] == 0.123
    assert dataframe.iloc[0]["Normalized Importance"] == 0.679
    assert dataframe.iloc[0]["Contribution"] == 67.9


def test_no_drift_does_not_claim_a_largest_shifted_feature() -> None:
    """Identical-dataset feature tables return the explicit no-shift state."""
    import pandas as pd

    assert _largest_shifted_feature(pd.DataFrame([
        {"feature_name": "latency", "jensen_shannon_divergence": 0.0, "shifted": False}
    ])) == "None"


def test_dataset_cards_use_actual_evaluation_train_test_counts() -> None:
    """Dataset cards replace unavailable split text when reports provide counts."""
    statistics = DatasetStatistics(2, 4, {}, {"latency": 0, "tokens": 0}, "summary")
    evaluation = {"evaluations": [{"train_size": 3, "test_size": 1}]}

    values = build_dataset_card_values(statistics, _dataset(), evaluation)

    assert values["Train/Test split"] == "3/1 (75% / 25%)"
    assert build_feature_importance_reports([{"model_name": "Model", "ranking": []}])


def test_confusion_observations_are_derived_from_matrix_values() -> None:
    """Observations identify the dynamic confusion and class recall extremes."""
    observations = build_confusion_matrix_observations(
        np.asarray([[8, 2], [1, 9]]), ["success", "timeout"]
    )

    assert "success predicted as timeout (2 sessions)" in observations[0]
    assert "timeout (90% recall)" in observations[1]
    assert "success (80% recall)" in observations[2]


def test_analytics_summary_uses_existing_reports_without_recalculation() -> None:
    """Top-level cards render existing metric, drift, and anomaly values."""
    statistics = DatasetStatistics(4, 2, {}, {"latency": 0, "tokens": 0}, "summary")
    evaluation = {
        "best_model": {"model_name": "Logistic Regression"},
        "evaluations": [{
            "model_name": "Logistic Regression",
            "metrics": {"accuracy": 0.875},
            "train_size": 3,
            "test_size": 1,
        }],
    }
    drift = {"summary": {"shifted_feature_count": 2}}
    anomaly = {"anomaly_count": 1}

    summary = build_analytics_summary(statistics, _dataset(), evaluation, drift, anomaly)

    assert summary["Best model"] == "Logistic Regression"
    assert summary["Accuracy"] == "87.5%"
    assert summary["Drift status"] == "2 shifted features"
    assert summary["Anomalies"] == "1"
    assert summary["Train/Test split"] == "3/1 (75% / 25%)"


def test_anomaly_summary_excludes_noise_from_cluster_count() -> None:
    """Dashboard cluster totals count DBSCAN clusters, not the noise label."""
    summary, _ = build_anomaly_summary(
        {
            "anomaly_count": 2,
            "noise_point_count": 4,
            "unknown_failure_count": 4,
            "cluster_statistics": [
                {"cluster_id": -1, "session_count": 4, "is_noise": True},
                {"cluster_id": 0, "session_count": 8, "is_noise": False},
            ],
        }
    )

    assert summary["Anomalies"] == 2
    assert summary["Noise points"] == 4
    assert summary["Cluster count"] == 1


def test_model_evaluation_highlight_marks_only_recommended_model() -> None:
    """Recommended-model styling remains a UI-only dataframe transformation."""
    import pandas as pd

    dataframe = pd.DataFrame(
        {"Model": ["Logistic Regression", "Random Forest"], "Accuracy": [0.9, 0.8]}
    )
    styled = style_model_evaluation_dataframe(dataframe, "Logistic Regression")
    context = styled._compute().ctx

    assert ("background-color", "#173d31") in context[(0, 0)]
    assert (1, 0) not in context


def test_model_insights_use_only_the_existing_importance_ranking() -> None:
    """Insight text remains a deterministic summary of supplied importance data."""
    insights = build_model_insights(
        {
            "ranking": [
                {"feature_name": "failed_llm_calls", "normalized_score": 0.5},
                {"feature_name": "json_errors_detected", "normalized_score": 0.3},
                {"feature_name": "failure_events", "normalized_score": 0.1},
            ]
        }
    )

    assert insights[1:4] == ["failed_llm_calls", "json_errors_detected", "failure_events"]
    assert insights[-1] == "The top three signals account for 90% of normalized importance."


def test_confusion_matrix_labels_use_compact_display_aliases() -> None:
    """Dense matrix labels are shortened without changing source class values."""
    assert format_confusion_matrix_label("context_window_overflow") == "ctx\nwindow"
    assert format_confusion_matrix_label("speech_to_text_failure") == "speech\nto_text"
    assert format_confusion_matrix_label("success") == "success"
