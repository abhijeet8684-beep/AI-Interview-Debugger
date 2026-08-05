"""ML analytics visualization helpers for the Streamlit dashboard.

This module contains pure data preparation helpers and plotting utilities used by
app/app.py to render ML analytics reports without creating new ML logic.

Helpers in this module are intentionally limited to JSON parsing, dataset
summary extraction, and chart assembly. The underlying ML modules remain frozen.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.dataset_loader import DatasetLoader, DatasetStatistics, MLDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "synthetic" / "interview_sessions.jsonl"

EvaluationReport = Dict[str, Any]
FeatureImportanceReport = Dict[str, Any]
DriftReport = Dict[str, Any]
AnomalyReport = Dict[str, Any]


def load_json_report(data: bytes) -> Optional[Dict[str, Any]]:
    """Parse a JSON report from raw uploaded bytes.

    Returns None when the payload is empty or invalid.
    """
    if not data:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_dataset_statistics(path: Optional[Path] = None) -> Tuple[DatasetStatistics, MLDataset]:
    """Load dataset statistics from a JSONL or CSV source.

    When no path is provided, the repository's synthetic sample dataset is used.
    """
    source_path = path or DEFAULT_DATASET_PATH
    loader = DatasetLoader()
    dataset = loader.load(source_path)
    return loader.statistics(dataset), dataset


def build_dataset_card_values(
    stats: DatasetStatistics,
    dataset: MLDataset,
    evaluation_report: Optional[EvaluationReport] = None,
) -> Dict[str, Any]:
    """Build dataset overview values for dashboard cards."""
    failure_categories = {label for label in dataset.labels.failure_categories if label}
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    if evaluation_report:
        first_evaluation = _first_evaluation(evaluation_report)
        if first_evaluation:
            train_size = first_evaluation.get("train_size")
            test_size = first_evaluation.get("test_size")
    card_values: Dict[str, Any] = {
        "Number of sessions": stats.row_count,
        "Number of features": stats.feature_count,
        "Failure categories": len(failure_categories),
        "Train/Test split": _format_train_test_split(train_size, test_size),
        "Missing values": sum(stats.missing_values.values()),
        "Dataset summary": stats.summary,
    }
    return card_values


def _format_train_test_split(train_size: Optional[int], test_size: Optional[int]) -> str:
    if train_size is None or test_size is None:
        return "Unavailable"
    total = train_size + test_size
    if total <= 0:
        return "Unavailable"
    ratio = int(round(100 * train_size / total))
    return f"{train_size}/{test_size} ({ratio}% / {100 - ratio}%)"


def _first_evaluation(report: EvaluationReport) -> Optional[Dict[str, Any]]:
    if not report:
        return None
    evaluations = report.get("evaluations")
    if isinstance(evaluations, list) and evaluations:
        return evaluations[0]
    if report.get("model_name") and report.get("metrics"):
        return report
    return None


def build_evaluation_dataframe(reports: List[EvaluationReport]) -> pd.DataFrame:
    """Build a comparison dataframe from one or more evaluation reports."""
    rows: List[Dict[str, Any]] = []
    for report in reports:
        for evaluation in _iter_evaluations(report):
            metrics = evaluation.get("metrics", {})
            rows.append(
                {
                    "Model": evaluation.get("model_name", "Unknown"),
                    "Accuracy": metrics.get("accuracy"),
                    "Precision": metrics.get("precision"),
                    "Recall": metrics.get("recall"),
                    "Macro F1": metrics.get("macro_f1"),
                    "Weighted F1": metrics.get("weighted_f1"),
                    "Micro F1": metrics.get("micro_f1"),
                }
            )
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=["Macro F1", "Accuracy"], ascending=[False, False])
    return dataframe


def _iter_evaluations(report: EvaluationReport) -> Iterable[Dict[str, Any]]:
    evaluations = report.get("evaluations")
    if isinstance(evaluations, list):
        return evaluations
    if report.get("model_name") and report.get("metrics"):
        return [report]
    return []


def evaluation_model_options(reports: List[EvaluationReport]) -> List[str]:
    """Return available model names from evaluation reports."""
    names = [evaluation.get("model_name", "Unknown") for report in reports for evaluation in _iter_evaluations(report)]
    return sorted(dict.fromkeys(names))


def get_confusion_matrix(report: EvaluationReport, model_name: str) -> Tuple[np.ndarray, List[str]]:
    """Get a confusion matrix and class labels for a selected model."""
    for evaluation in _iter_evaluations(report):
        if evaluation.get("model_name") == model_name:
            matrix = np.asarray(evaluation.get("confusion_matrix", []), dtype=int)
            labels = [str(label) for label in evaluation.get("class_labels", [])]
            return matrix, labels
    return np.empty((0, 0), dtype=int), []


def build_confusion_matrix_figure(matrix: np.ndarray, labels: List[str]) -> plt.Figure:
    """Render a confusion matrix figure using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if matrix.size == 0:
        ax.text(0.5, 0.5, "No confusion matrix data available", ha="center", va="center")
        ax.axis("off")
        return fig
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def build_feature_importance_reports(reports: List[FeatureImportanceReport]) -> List[FeatureImportanceReport]:
    """Return only valid feature importance reports from uploaded files."""
    valid_reports = []
    for report in reports:
        if report.get("model_name") and isinstance(report.get("ranking"), list):
            valid_reports.append(report)
    return valid_reports


def feature_importance_model_options(reports: List[FeatureImportanceReport]) -> List[str]:
    return sorted(dict.fromkeys(report["model_name"] for report in reports if report.get("model_name")))


def feature_importance_dataframe(report: FeatureImportanceReport, top_n: int = 15) -> pd.DataFrame:
    """Build a dataframe of top-n feature importance values for a model."""
    ranking = report.get("ranking", [])[:top_n]
    rows = [
        {
            "Feature": item.get("feature_name"),
            "Importance": item.get("importance_score"),
            "Normalized": item.get("normalized_score"),
            "Rank": item.get("rank"),
        }
        for item in ranking
    ]
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=["Importance", "Feature"], ascending=[False, True])
    return dataframe


def feature_importance_figure(dataframe: pd.DataFrame) -> plt.Figure:
    """Render a horizontal bar chart for feature importance."""
    fig, ax = plt.subplots(figsize=(8, min(6, 0.35 * len(dataframe) + 1)))
    if dataframe.empty:
        ax.text(0.5, 0.5, "No feature importance data available", ha="center", va="center")
        ax.axis("off")
        return fig
    df = dataframe.copy()
    df = df.sort_values(by="Importance", ascending=True)
    ax.barh(df["Feature"], df["Importance"], color="#2b8cc4")
    ax.set_xlabel("Importance score")
    ax.set_title("Top Feature Importance")
    fig.tight_layout()
    return fig


def build_drift_summary(report: DriftReport) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Construct drift summary cards and dataframes for display."""
    summary = report.get("summary", {})
    feature_drift = pd.DataFrame(report.get("feature_drift", []))
    label_drift = pd.DataFrame(report.get("label_drift", []))
    drift_values = {
        "Overall drift score": summary.get("overall_drift_score", 0.0),
        "Largest shifted feature": _largest_shifted_feature(feature_drift),
        "Average drift": summary.get("average_drift", 0.0),
    }
    return drift_values, feature_drift, label_drift


def _largest_shifted_feature(feature_drift: pd.DataFrame) -> str:
    if feature_drift.empty:
        return "None"
    if "jensen_shannon_divergence" not in feature_drift.columns:
        return "Unknown"
    row = feature_drift.loc[feature_drift["jensen_shannon_divergence"].idxmax()]
    return f"{row.get('feature_name', 'Unknown')} ({row.get('jensen_shannon_divergence', 0.0):.2f})"


def style_feature_drift_dataframe(feature_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    if feature_df.empty:
        return feature_df.style
    renamed = feature_df.rename(
        columns={
            "feature_name": "Feature",
            "jensen_shannon_divergence": "JS Divergence",
            "population_stability_index": "PSI",
            "shifted": "Shifted",
        }
    )
    return renamed.style.format(
        {"JS Divergence": "{:.3f}", "PSI": "{:.3f}"}
    ).applymap(_drift_color, subset=["JS Divergence"]).applymap(_shifted_color, subset=["Shifted"])


def _drift_color(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric < 0.1:
        return "background-color: #e6f4ea"
    if numeric < 0.3:
        return "background-color: #fff4cc"
    return "background-color: #fdecea"


def _shifted_color(value: Any) -> str:
    if value is True or str(value).lower() == "true":
        return "background-color: #fdecea"
    return "background-color: #e6f4ea"


def style_label_drift_dataframe(label_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    if label_df.empty:
        return label_df.style
    renamed = label_df.rename(
        columns={
            "label_group": "Label group",
            "label": "Label",
            "baseline_percentage": "Baseline %",
            "current_percentage": "Current %",
            "percentage_change": "Change %",
        }
    )
    return renamed.style.format(
        {"Baseline %": "{:.1f}", "Current %": "{:.1f}", "Change %": "{:+.1f}"}
    ).applymap(_label_drift_color, subset=["Change %"])


def _label_drift_color(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if abs(numeric) < 2.0:
        return "background-color: #e6f4ea"
    if abs(numeric) < 5.0:
        return "background-color: #fff4cc"
    return "background-color: #fdecea"


def build_anomaly_summary(report: AnomalyReport) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Build anomaly summary metrics and a cluster statistics dataframe."""
    summary = {
        "Anomalies": report.get("anomaly_count", 0),
        "Noise points": report.get("noise_point_count", 0),
        "Unknown failure candidates": report.get("unknown_failure_count", 0),
        "Cluster groups": len(report.get("cluster_statistics", [])),
    }
    cluster_df = pd.DataFrame(report.get("cluster_statistics", []))
    if not cluster_df.empty:
        cluster_df = cluster_df.rename(
            columns={"cluster_id": "Cluster ID", "session_count": "Session count", "is_noise": "Noise"}
        )
    return summary, cluster_df


def write_temp_jsonl(data: bytes, suffix: str) -> Path:
    """Write bytes to a temporary file for dataset loading."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(data)
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def load_dataset_from_bytes(data: bytes, suffix: str) -> Tuple[DatasetStatistics, MLDataset]:
    temp_path = write_temp_jsonl(data, suffix)
    try:
        return load_dataset_statistics(temp_path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def create_download_payload(report: Optional[Dict[str, Any]]) -> Optional[str]:
    if report is None:
        return None
    return json.dumps(report, indent=2, sort_keys=True)


def render_ml_analytics() -> None:
    """Render the ML analytics tab for the Streamlit dashboard."""
    import streamlit as st

    st.header("📊 ML Analytics")
    st.write(
        "Visualize offline ML analytics reports for dataset overview, model evaluation, "
        "feature importance, drift detection, and anomaly analysis. Upload report "
        "JSON files to populate the dashboard sections."
    )

    with st.expander("Load offline ML reports and dataset", expanded=True):
        dataset_col, report_col = st.columns([2, 3])
        dataset_file = dataset_col.file_uploader(
            "Dataset file",
            type=["jsonl", "ndjson", "csv"],
            key="ml_dataset",
            help="Upload a synthetic dataset or use the built-in sample dataset.",
        )
        evaluation_files = report_col.file_uploader(
            "Evaluation report JSON",
            type=["json"],
            accept_multiple_files=True,
            key="ml_evaluation_reports",
            help="Upload one or more model evaluation report JSON files.",
        )
        importance_files = report_col.file_uploader(
            "Feature importance JSON",
            type=["json"],
            accept_multiple_files=True,
            key="ml_importance_reports",
            help="Upload feature importance reports for one or more models.",
        )
        drift_file = report_col.file_uploader(
            "Drift report JSON",
            type=["json"],
            key="ml_drift_report",
            help="Upload a drift detection report JSON file.",
        )
        anomaly_file = report_col.file_uploader(
            "Anomaly report JSON",
            type=["json"],
            key="ml_anomaly_report",
            help="Upload an anomaly detection report JSON file.",
        )

    dataset_statistics: Optional[DatasetStatistics] = None
    dataset: Optional[MLDataset] = None
    dataset_error: Optional[str] = None
    try:
        if dataset_file is not None:
            suffix = Path(dataset_file.name).suffix or ".jsonl"
            dataset_statistics, dataset = load_dataset_from_bytes(dataset_file.getvalue(), suffix)
        else:
            dataset_statistics, dataset = load_dataset_statistics()
    except Exception as error:
        dataset_error = str(error)

    evaluation_reports: List[EvaluationReport] = []
    for uploaded in evaluation_files or []:
        report = load_json_report(uploaded.getvalue())
        if report is not None:
            evaluation_reports.append(report)

    importance_reports: List[FeatureImportanceReport] = []
    for uploaded in importance_files or []:
        report = load_json_report(uploaded.getvalue())
        if report is not None:
            importance_reports.append(report)

    drift_report = load_json_report(drift_file.getvalue()) if drift_file is not None else None
    anomaly_report = load_json_report(anomaly_file.getvalue()) if anomaly_file is not None else None

    st.subheader("Dataset Overview")
    if dataset_error:
        st.error(f"Unable to load dataset: {dataset_error}")
    elif dataset_statistics is None or dataset is None:
        st.info("No dataset is currently available for overview.")
    else:
        overview_values = build_dataset_card_values(
            dataset_statistics, dataset, evaluation_reports[0] if evaluation_reports else None
        )
        metrics = st.columns(4)
        metrics[0].metric("Sessions", overview_values["Number of sessions"])
        metrics[1].metric("Features", overview_values["Number of features"])
        metrics[2].metric("Failure categories", overview_values["Failure categories"])
        metrics[3].metric("Missing values", overview_values["Missing values"])
        st.markdown(f"**Train/Test split:** {overview_values['Train/Test split']}")
        st.caption(overview_values["Dataset summary"])
        missing_df = pd.DataFrame(
            sorted(dataset_statistics.missing_values.items(), key=lambda item: item[1], reverse=True),
            columns=["Feature", "Missing values"],
        )
        if not missing_df.empty:
            st.write("##### Missing values by feature")
            st.dataframe(missing_df.head(15), use_container_width=True)

    st.markdown("---")
    st.subheader("Model Evaluation")
    if not evaluation_reports:
        st.info("Upload at least one evaluation report JSON file to visualize model metrics.")
    else:
        eval_df = build_evaluation_dataframe(evaluation_reports)
        if eval_df.empty:
            st.info("Evaluation reports were loaded but did not contain valid model summaries.")
        else:
            st.write("##### Comparison of offline model evaluation metrics")
            st.dataframe(eval_df, use_container_width=True)
            model_names = evaluation_model_options(evaluation_reports)
            selected_model = st.selectbox(
                "Select model for confusion matrix", model_names, index=0
            )
            confusion_matrix, class_labels = np.empty((0, 0), dtype=int), []
            for report in evaluation_reports:
                matrix, labels = get_confusion_matrix(report, selected_model)
                if matrix.size:
                    confusion_matrix, class_labels = matrix, labels
                    break
            st.write("##### Confusion matrix")
            st.pyplot(build_confusion_matrix_figure(confusion_matrix, class_labels))

    st.markdown("---")
    st.subheader("Feature Importance")
    valid_importance_reports = build_feature_importance_reports(importance_reports)
    if not valid_importance_reports:
        st.info("Upload feature importance JSON reports to compare model explainability.")
    else:
        importance_options = feature_importance_model_options(valid_importance_reports)
        selected_feature_model = st.selectbox(
            "Select model for feature importance", importance_options, index=0
        )
        model_report = next(
            (report for report in valid_importance_reports if report.get("model_name") == selected_feature_model),
            valid_importance_reports[0],
        )
        importance_df = feature_importance_dataframe(model_report, top_n=15)
        st.write("##### Top 15 features")
        st.dataframe(importance_df, use_container_width=True)
        st.pyplot(feature_importance_figure(importance_df))

    st.markdown("---")
    st.subheader("Drift Detection")
    if drift_report is None:
        st.info("Upload a drift report JSON file to display feature and label drift analytics.")
    else:
        drift_values, feature_drift_df, label_drift_df = build_drift_summary(drift_report)
        drift_cols = st.columns(3)
        drift_cols[0].metric("Overall drift score", f"{drift_values['Overall drift score']:.3f}")
        drift_cols[1].metric("Largest shifted feature", drift_values["Largest shifted feature"])
        drift_cols[2].metric("Average drift", f"{drift_values['Average drift']:.3f}")
        st.write("##### Feature drift")
        st.dataframe(style_feature_drift_dataframe(feature_drift_df), use_container_width=True)
        st.write("##### Label drift")
        st.dataframe(style_label_drift_dataframe(label_drift_df), use_container_width=True)

    st.markdown("---")
    st.subheader("Anomaly Detection")
    if anomaly_report is None:
        st.info("Upload an anomaly report JSON file to show anomaly and cluster summaries.")
    else:
        anomaly_summary, cluster_df = build_anomaly_summary(anomaly_report)
        anomaly_cols = st.columns(4)
        anomaly_cols[0].metric("Anomalies", anomaly_summary["Anomalies"])
        anomaly_cols[1].metric("Noise points", anomaly_summary["Noise points"])
        anomaly_cols[2].metric("Unknown failures", anomaly_summary["Unknown failure candidates"])
        anomaly_cols[3].metric("Cluster groups", anomaly_summary["Cluster groups"])
        if not cluster_df.empty:
            st.write("##### DBSCAN cluster statistics")
            st.dataframe(cluster_df, use_container_width=True)
        records = anomaly_report.get("records")
        if isinstance(records, list) and records:
            st.write("##### Anomaly records")
            st.dataframe(
                pd.DataFrame(records)[
                    ["session_id", "anomaly_score", "is_anomaly", "cluster_id", "is_noise", "is_unknown_failure"]
                ],
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Export ML Reports")
    eval_payload = create_download_payload(evaluation_reports[0] if evaluation_reports else None)
    importance_payload = create_download_payload(valid_importance_reports[0] if valid_importance_reports else None)
    drift_payload = create_download_payload(drift_report)
    anomaly_payload = create_download_payload(anomaly_report)
    export_cols = st.columns(4)
    export_cols[0].download_button(
        "Download evaluation report",
        data=eval_payload or "{}",
        file_name="evaluation_report.json",
        mime="application/json",
        disabled=eval_payload is None,
    )
    export_cols[1].download_button(
        "Download feature importance report",
        data=importance_payload or "{}",
        file_name="feature_importance_report.json",
        mime="application/json",
        disabled=importance_payload is None,
    )
    export_cols[2].download_button(
        "Download drift report",
        data=drift_payload or "{}",
        file_name="drift_report.json",
        mime="application/json",
        disabled=drift_payload is None,
    )
    export_cols[3].download_button(
        "Download anomaly report",
        data=anomaly_payload or "{}",
        file_name="anomaly_report.json",
        mime="application/json",
        disabled=anomaly_payload is None,
    )
