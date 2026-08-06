"""ML analytics visualization helpers for the Streamlit dashboard.

This module contains pure data preparation helpers and plotting utilities used by
app/app.py to render ML analytics reports without creating new ML logic.

Helpers in this module are intentionally limited to JSON parsing, dataset
summary extraction, and chart assembly. The underlying ML modules remain frozen.
"""
from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.anomaly_detection import AnomalyDetector
from ml.dataset_loader import DatasetLoader, DatasetStatistics, MLDataset
from ml.drift_detection import DriftDetector
from ml.evaluate_models import EvaluationReport as EvaluationReportObject, ModelEvaluator
from ml.feature_importance import FeatureImportanceAnalyzer, FeatureImportanceReport as FeatureImportanceReportObject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "synthetic" / "interview_sessions.jsonl"
DEFAULT_DATASET_KEY = "default_sample_dataset"

EvaluationReport = Dict[str, Any]
FeatureImportanceReport = Dict[str, Any]
DriftReport = Dict[str, Any]
AnomalyReport = Dict[str, Any]
ProgressCallback = Callable[[int, str], None]


def _card_container(st: Any) -> Any:
    """Return a bordered container when supported by the installed Streamlit."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _render_figure(st: Any, figure: plt.Figure) -> None:
    """Render a chart at available width with compatibility fallback."""
    try:
        st.pyplot(figure, use_container_width=True)
    except TypeError:
        st.pyplot(figure)


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


def format_confusion_matrix_label(label: str) -> str:
    """Return a compact, display-only label for dense confusion matrices."""
    normalized = label.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "context_window": "ctx\nwindow",
        "context_window_overflow": "ctx\nwindow",
        "db_failure": "db\nfailure",
        "database_failure": "db\nfailure",
        "eval_failure": "eval\nfailure",
        "evaluation_failure": "eval\nfailure",
        "json_invalid": "json\ninvalid",
        "invalid_json_response": "json\ninvalid",
        "json_validation_error": "json\ninvalid",
        "llm_timeout": "llm\ntimeout",
        "network_api": "network\napi",
        "network_api_failure": "network\napi",
        "retrieval": "retrieval",
        "retrieval_failure": "retrieval",
        "speech_to_text": "speech\nto_text",
        "speech_to_text_failure": "speech\nto_text",
        "successful_interview": "success",
        "tool_timeout": "tool\ntimeout",
    }
    if normalized in aliases:
        return aliases[normalized]
    return normalized.replace("_", "\n") if len(normalized) > 12 else normalized


def summarize_confusion_matrix(matrix: np.ndarray, labels: List[str]) -> str:
    """Summarize the largest observed off-diagonal confusion deterministically."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not labels:
        return "No valid confusion matrix is available for interpretation."
    largest_count = 0
    largest_pair: Optional[Tuple[int, int]] = None
    for actual_index in range(matrix.shape[0]):
        for predicted_index in range(matrix.shape[1]):
            value = int(matrix[actual_index, predicted_index])
            if actual_index != predicted_index and value > largest_count:
                largest_count = value
                largest_pair = (actual_index, predicted_index)
    if largest_pair is None:
        return "No class confusions were observed in this evaluation holdout."
    actual_index, predicted_index = largest_pair
    return (
        f"The classifier most frequently confuses {labels[actual_index]} with "
        f"{labels[predicted_index]} ({largest_count} sessions), indicating "
        "overlapping telemetry features in the evaluated holdout."
    )


def build_confusion_matrix_observations(matrix: np.ndarray, labels: List[str]) -> List[str]:
    """Return concise, data-derived observations for a confusion matrix.

    The observations are presentation-only summaries of the selected model's
    holdout results; they do not alter model evaluation or ranking.
    """
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or len(labels) != matrix.shape[0]
    ):
        return ["No valid confusion matrix is available for observations."]

    row_totals = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        row_totals,
        out=np.zeros(len(labels), dtype=float),
        where=row_totals > 0,
    )
    best_index = int(np.argmax(recalls))
    worst_index = int(np.argmin(recalls))
    observations = [
        f"Best classified class: {labels[best_index]} "
        f"({recalls[best_index]:.0%} recall).",
        f"Lowest recall class: {labels[worst_index]} "
        f"({recalls[worst_index]:.0%} recall).",
    ]
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    largest_error = int(off_diagonal.max())
    if largest_error > 0:
        actual_index, predicted_index = np.unravel_index(
            off_diagonal.argmax(), off_diagonal.shape
        )
        observations.insert(
            0,
            f"Most confused classes: {labels[actual_index]} predicted as "
            f"{labels[predicted_index]} ({largest_error} sessions).",
        )
    else:
        observations.insert(0, "No cross-class confusions were observed in this holdout.")
    return observations


def get_best_model_evaluation(report: EvaluationReport) -> Optional[Dict[str, Any]]:
    """Return the evaluated metrics for the report's selected best model."""
    best_model = report.get("best_model", {}).get("model_name")
    if not best_model:
        return None
    return next(
        (item for item in _iter_evaluations(report) if item.get("model_name") == best_model),
        None,
    )


def build_confusion_matrix_figure(matrix: np.ndarray, labels: List[str]) -> plt.Figure:
    """Render a compact, theme-compatible confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    if matrix.size == 0:
        ax.text(
            0.5,
            0.5,
            "No confusion matrix data available",
            ha="center",
            va="center",
            color="#e5e7eb",
            fontsize=11,
        )
        ax.axis("off")
        return fig
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.ax.tick_params(colors="#e5e7eb", labelsize=9)
    colorbar.outline.set_edgecolor("#9ca3af")
    display_labels = [format_confusion_matrix_label(label) for label in labels]
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, fontsize=8, color="#e5e7eb")
    ax.set_yticklabels(display_labels, fontsize=8, color="#e5e7eb")
    ax.set_xlabel("Predicted label", fontsize=11, color="#e5e7eb")
    ax.set_ylabel("Actual label", fontsize=11, color="#e5e7eb")
    ax.tick_params(colors="#e5e7eb")
    for spine in ax.spines.values():
        spine.set_edgecolor("#9ca3af")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#111827", fontsize=10)
    fig.tight_layout(pad=0.5)
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
            "Top 5": int(item.get("rank", 0)) <= 5,
            "Top 3": int(item.get("rank", 0)) <= 3,
            "Rank": item.get("rank"),
            "Feature Name": item.get("feature_name"),
            "Raw Importance": round(float(item.get("importance_score", 0.0)), 3),
            "Normalized Importance": round(float(item.get("normalized_score", 0.0)), 3),
            "Contribution": round(float(item.get("normalized_score", 0.0)) * 100, 1),
        }
        for item in ranking
    ]
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=["Rank", "Feature Name"], ascending=[True, True])
    return dataframe


def feature_importance_figure(dataframe: pd.DataFrame) -> plt.Figure:
    """Render a compact, theme-compatible feature-importance chart."""
    fig, ax = plt.subplots(figsize=(9.0, min(4.6, 0.28 * len(dataframe) + 1.0)))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    if dataframe.empty:
        ax.text(
            0.5,
            0.5,
            "No feature importance data available",
            ha="center",
            va="center",
            color="#e5e7eb",
            fontsize=11,
        )
        ax.axis("off")
        return fig
    df = dataframe.copy()
    df = df.sort_values(by="Raw Importance", ascending=True)
    bars = ax.barh(df["Feature Name"], df["Raw Importance"], color="#2b8cc4")
    ax.set_xlabel("Raw importance score", fontsize=10, color="#e5e7eb")
    ax.set_title("Top Feature Importance", fontsize=12, color="#f9fafb", pad=5)
    ax.tick_params(axis="both", colors="#e5e7eb", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#9ca3af")
    ax.bar_label(bars, fmt="%.3f", padding=2, color="#e5e7eb", fontsize=9)
    fig.tight_layout(pad=0.5)
    return fig


def build_model_insights(report: Optional[FeatureImportanceReport]) -> List[str]:
    """Build concise, deterministic insights from an importance ranking."""
    if not report:
        return []
    ranking = report.get("ranking", [])
    top_features = [str(item.get("feature_name")) for item in ranking[:3] if item.get("feature_name")]
    if not top_features:
        return []
    contribution = sum(float(item.get("normalized_score", 0.0)) for item in ranking[:3])
    return [
        "Primary prediction drivers:",
        *top_features,
        f"The top three signals account for {contribution:.0%} of normalized importance.",
    ]


def style_model_evaluation_dataframe(
    dataframe: pd.DataFrame, best_model: Optional[str]
) -> pd.io.formats.style.Styler:
    """Apply subtle display-only highlighting to the recommended model row."""
    if dataframe.empty or not best_model:
        return dataframe.style

    def highlight(row: pd.Series) -> List[str]:
        color = (
            "background-color: #173d31; color: #f3f4f6;"
            if row["Model"] == best_model
            else ""
        )
        return [color] * len(row)

    return dataframe.style.apply(highlight, axis=1)


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
    if "shifted" not in feature_drift.columns or not feature_drift["shifted"].any():
        return "None"
    shifted = feature_drift[feature_drift["shifted"]]
    row = shifted.loc[shifted["jensen_shannon_divergence"].idxmax()]
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
    styler = renamed.style.format({"JS Divergence": "{:.3f}", "PSI": "{:.3f}"})
    try:
        # Preferred styling: applymap if available.
        if hasattr(styler, "applymap"):
            styler = styler.applymap(_drift_color, subset=["JS Divergence"]).applymap(
                _shifted_color, subset=["Shifted"]
            )
        else:
            # Fallback: use apply on the column series to return per-cell styles.
            styler = styler.apply(lambda s: s.map(_drift_color), subset=["JS Divergence"])
            styler = styler.apply(lambda s: s.map(_shifted_color), subset=["Shifted"])
    except Exception:
        # If any styling fails (pandas incompatibility), return an unstyled styler with formatting.
        try:
            return renamed.style.format({"JS Divergence": "{:.3f}", "PSI": "{:.3f}"})
        except Exception:
            return renamed.style
    return styler


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
    styler = renamed.style.format({"Baseline %": "{:.1f}", "Current %": "{:.1f}", "Change %": "{:+.1f}"})
    try:
        if hasattr(styler, "applymap"):
            styler = styler.applymap(_label_drift_color, subset=["Change %"])
        else:
            styler = styler.apply(lambda s: s.map(_label_drift_color), subset=["Change %"])
    except Exception:
        try:
            return renamed.style.format({"Baseline %": "{:.1f}", "Current %": "{:.1f}", "Change %": "{:+.1f}"})
        except Exception:
            return renamed.style
    return styler


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
        "Cluster count": sum(
            not bool(item.get("is_noise", False))
            for item in report.get("cluster_statistics", [])
        ),
    }
    cluster_df = pd.DataFrame(report.get("cluster_statistics", []))
    if not cluster_df.empty:
        cluster_df = cluster_df.rename(
            columns={"cluster_id": "Cluster ID", "session_count": "Session count", "is_noise": "Noise"}
        )
    return summary, cluster_df


def build_analytics_summary(
    stats: Optional[DatasetStatistics],
    dataset: Optional[MLDataset],
    evaluation_report: Optional[EvaluationReport],
    drift_report: Optional[DriftReport],
    anomaly_report: Optional[AnomalyReport],
) -> Dict[str, str]:
    """Build compact display values for the analytics-page summary panel."""
    cards = (
        build_dataset_card_values(stats, dataset, evaluation_report)
        if stats is not None and dataset is not None
        else {}
    )
    best_evaluation = get_best_model_evaluation(evaluation_report or {})
    metrics = best_evaluation.get("metrics", {}) if best_evaluation else {}
    shifted_count = (drift_report or {}).get("summary", {}).get("shifted_feature_count", 0)
    drift_status = "Unavailable" if drift_report is None else (
        "No significant drift" if not shifted_count else f"{shifted_count} shifted features"
    )
    return {
        "Sessions analysed": str(cards.get("Number of sessions", "Unavailable")),
        "Features": str(cards.get("Number of features", "Unavailable")),
        "Best model": (
            str(best_evaluation.get("model_name", "Unavailable"))
            if best_evaluation
            else "Unavailable"
        ),
        "Accuracy": (
            f"{float(metrics.get('accuracy', 0.0)):.1%}"
            if best_evaluation
            else "Unavailable"
        ),
        "Drift status": drift_status,
        "Anomalies": str((anomaly_report or {}).get("anomaly_count", "Unavailable")),
        "Train/Test split": str(cards.get("Train/Test split", "Unavailable")),
    }


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


def dataset_cache_key(file_name: Optional[str], data: Optional[bytes]) -> str:
    """Generate a stable hash key for an uploaded dataset file.

    The key is used to cache ML analytics results for the same dataset content.
    """
    hasher = sha256()
    hasher.update((file_name or DEFAULT_DATASET_KEY).encode("utf-8"))
    if data is not None:
        hasher.update(data)
    return hasher.hexdigest()


def _report_progress(callback: Optional[ProgressCallback], percent: int, message: str) -> None:
    if callback is None:
        return
    callback(percent, message)


def _feature_label_array(dataset: MLDataset, label_name: str = "failure_category") -> np.ndarray:
    label_mapping = {
        "root_cause": "root_causes",
        "root_causes": "root_causes",
        "affected_component": "affected_components",
        "affected_components": "affected_components",
        "failure_category": "failure_categories",
        "failure_categories": "failure_categories",
    }
    mapped_name = label_mapping.get(label_name)
    if mapped_name is None or mapped_name not in dataset.labels.__dict__:
        raise ValueError(f"Unsupported label name: {label_name}")
    labels = getattr(dataset.labels, mapped_name)
    if labels is None:
        raise ValueError(f"Dataset is missing '{mapped_name}' labels.")
    return np.asarray([str(label) for label in labels], dtype=str)


def _make_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(random_state=random_state, max_iter=1000),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "classifier",
                    RandomForestClassifier(random_state=random_state, n_estimators=100),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=random_state, n_estimators=100),
                ),
            ]
        ),
    }


def run_ml_analytics(
    dataset: MLDataset,
    baseline_dataset: MLDataset,
    random_state: int = 42,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Execute the ML analytics workflow and return serializable reports."""
    _report_progress(progress_callback, 5, "Validating dataset and labels...")
    model_evaluator = ModelEvaluator(random_state=random_state)
    _report_progress(progress_callback, 20, "Training and evaluating models...")
    evaluation_report: EvaluationReportObject = model_evaluator.evaluate(dataset)
    _report_progress(progress_callback, 45, "Computing feature importance...")
    feature_analyzer = FeatureImportanceAnalyzer(random_state=random_state)
    labels = _feature_label_array(dataset)
    feature_importance_reports: List[FeatureImportanceReportObject] = []
    for _, pipeline in _make_model_pipelines(random_state).items():
        fitted_pipeline = pipeline.fit(dataset.features, labels)
        try:
            report = feature_analyzer.analyze(
                fitted_pipeline,
                tuple(dataset.feature_names),
                dataset.features,
                labels,
            )
            feature_importance_reports.append(report)
        except Exception:
            continue
    _report_progress(progress_callback, 65, "Running drift detection...")
    drift_detector = DriftDetector()
    drift_report = drift_detector.compare(baseline_dataset, dataset)
    _report_progress(progress_callback, 85, "Running anomaly detection...")
    anomaly_detector = AnomalyDetector(random_state=random_state)
    anomaly_report = anomaly_detector.analyze(dataset)
    _report_progress(progress_callback, 100, "ML analytics complete.")
    return {
        "evaluation_report": evaluation_report.to_dict(),
        "feature_importance_reports": [report.to_dict() for report in feature_importance_reports],
        "drift_report": drift_report.to_dict(),
        "anomaly_report": anomaly_report.to_dict(),
    }


def create_download_payload(report: Optional[Dict[str, Any]]) -> Optional[str]:
    if report is None:
        return None
    return json.dumps(report, indent=2, sort_keys=True)


def _render_dataset_overview(
    st: Any,
    dataset_statistics: Optional[DatasetStatistics],
    dataset: Optional[MLDataset],
    dataset_error: Optional[str],
    evaluation_report: Optional[EvaluationReport],
) -> None:
    st.subheader("Dataset Overview")
    if dataset_error:
        st.error(f"Unable to load dataset: {dataset_error}")
        return
    if dataset_statistics is None or dataset is None:
        st.info("No dataset is currently available for overview. Upload a dataset and press Run ML Analytics.")
        return

    overview_values = build_dataset_card_values(
        dataset_statistics, dataset, evaluation_report
    )
    with _card_container(st):
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
    total_missing = int(missing_df["Missing values"].sum()) if not missing_df.empty else 0
    if total_missing == 0:
        st.success("No missing values detected")
        st.caption(f"{overview_values['Number of features']}/{overview_values['Number of features']} features complete")
    else:
        st.write("##### Missing values by feature")
        st.dataframe(missing_df.head(15), use_container_width=True)


def _render_analytics_summary(
    st: Any,
    dataset_statistics: Optional[DatasetStatistics],
    dataset: Optional[MLDataset],
    evaluation_report: Optional[EvaluationReport],
    drift_report: Optional[DriftReport],
    anomaly_report: Optional[AnomalyReport],
) -> None:
    """Render the compact, report-derived overview at the top of analytics."""
    summary = build_analytics_summary(
        dataset_statistics,
        dataset,
        evaluation_report,
        drift_report,
        anomaly_report,
    )
    with _card_container(st):
        st.write("##### Analytics summary")
        first_row = st.columns(4)
        first_row[0].metric("Sessions analysed", summary["Sessions analysed"])
        first_row[1].metric("Features", summary["Features"])
        first_row[2].metric("Best model", summary["Best model"])
        first_row[3].metric("Accuracy", summary["Accuracy"])
        second_row = st.columns(3)
        second_row[0].metric("Drift status", summary["Drift status"])
        second_row[1].metric("Anomalies", summary["Anomalies"])
        second_row[2].metric("Train/Test split", summary["Train/Test split"])


def _render_model_evaluation(
    st: Any,
    evaluation_report: Optional[EvaluationReport],
    importance_reports: Optional[List[FeatureImportanceReport]] = None,
) -> None:
    st.subheader("Model Evaluation")
    st.caption("Compares offline models used to classify interview failure categories.")
    if evaluation_report is None:
        st.info("Run ML Analytics to generate model evaluation metrics.")
        return

    reports = [evaluation_report]
    eval_df = build_evaluation_dataframe(reports)
    if eval_df.empty:
        st.info("No valid evaluation metrics were generated for this dataset.")
        return

    best_evaluation = get_best_model_evaluation(evaluation_report)
    best_model = best_evaluation.get("model_name") if best_evaluation else None
    if best_evaluation:
        best_metrics = best_evaluation.get("metrics", {})
        with _card_container(st):
            st.write("##### Recommended model")
            metric_columns = st.columns(5)
            metric_columns[0].metric("Model", str(best_model))
            metric_columns[1].metric("Accuracy", f"{float(best_metrics.get('accuracy', 0.0)):.1%}")
            metric_columns[2].metric("Precision", f"{float(best_metrics.get('precision', 0.0)):.1%}")
            metric_columns[3].metric("Recall", f"{float(best_metrics.get('recall', 0.0)):.1%}")
            metric_columns[4].metric("F1 Score", f"{float(best_metrics.get('f1_score', 0.0)):.1%}")
            st.info(
                f"{best_model} achieved the highest overall performance and is "
                "recommended for deployment on this dataset."
            )
        insights = build_model_insights((importance_reports or [None])[0])
        if insights:
            with _card_container(st):
                st.write("##### Model Insights")
                st.write(f"- **{insights[0]}**")
                for feature_name in insights[1:-1]:
                    st.write(f"  - {feature_name}")
                st.caption(insights[-1])
    st.write("##### Comparison of offline model evaluation metrics")
    st.dataframe(
        style_model_evaluation_dataframe(eval_df, best_model),
        use_container_width=True,
        hide_index=True,
    )
    model_names = evaluation_model_options(reports)
    selected_model = st.selectbox("Select model for confusion matrix", model_names, index=0)
    confusion_matrix, class_labels = get_confusion_matrix(reports[0], selected_model)
    matrix_column, observation_column = st.columns([3, 2])
    with matrix_column:
        st.write("##### Confusion matrix")
        _render_figure(st, build_confusion_matrix_figure(confusion_matrix, class_labels))
        st.caption(summarize_confusion_matrix(confusion_matrix, class_labels))
    with observation_column:
        with _card_container(st):
            st.write("##### Observations")
            for observation in build_confusion_matrix_observations(
                confusion_matrix, class_labels
            ):
                st.write(f"- {observation}")


def _render_feature_importance(st: Any, importance_reports: List[FeatureImportanceReport]) -> None:
    st.subheader("Feature Importance")
    st.caption("Ranks telemetry features by their influence on model predictions.")
    if not importance_reports:
        st.info("Run ML Analytics to generate feature importance reports.")
        return

    importance_options = feature_importance_model_options(importance_reports)
    selected_feature_model = st.selectbox(
        "Select model for feature importance", importance_options, index=0
    )
    model_report = next(
        (report for report in importance_reports if report.get("model_name") == selected_feature_model),
        importance_reports[0],
    )
    query = st.text_input("Search feature names", help="Filter the feature ranking by name.")
    importance_df = feature_importance_dataframe(model_report, top_n=15)
    if query:
        importance_df = importance_df[
            importance_df["Feature Name"].str.contains(query, case=False, na=False)
        ]
    st.write("##### Top 15 features")
    dataframe_options: Dict[str, Any] = {
        "use_container_width": True,
        "hide_index": True,
    }
    if hasattr(st, "column_config"):
        dataframe_options["column_config"] = {
            "Top 3": st.column_config.CheckboxColumn(
                "Top 3", help="Highlights the three leading features."
            ),
            "Top 5": st.column_config.CheckboxColumn(
                "Top 5", help="Marks the five leading features."
            ),
            "Contribution": st.column_config.ProgressColumn(
                "Contribution",
                help="Normalized importance contribution.",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        }
    st.dataframe(importance_df, **dataframe_options)
    _render_figure(st, feature_importance_figure(importance_df))


def _render_drift_detection(st: Any, drift_report: Optional[DriftReport]) -> None:
    st.subheader("Drift Detection")
    st.caption("Drift compares the uploaded dataset with the baseline reference dataset.")
    if drift_report is None:
        st.info("Run ML Analytics to generate drift detection analytics.")
        return

    drift_values, feature_drift_df, label_drift_df = build_drift_summary(drift_report)
    overall_drift = float(drift_values["Overall drift score"])
    if overall_drift == 0.0:
        st.success("Dataset matches baseline")
        st.caption("No shifted features detected.")
        with st.expander("Show detailed drift metrics"):
            st.dataframe(label_drift_df, use_container_width=True, hide_index=True)
        return

    drift_cols = st.columns(3)
    drift_cols[0].metric("Overall drift score", f"{drift_values['Overall drift score']:.3f}")
    shifted_features = feature_drift_df[
        feature_drift_df.get("shifted", pd.Series(dtype=bool)).astype(bool)
    ] if "shifted" in feature_drift_df else feature_drift_df
    if drift_values["Largest shifted feature"] == "None":
        st.success(
            "No significant drift detected. The uploaded dataset remains aligned "
            "with the baseline."
        )
    else:
        st.warning("Significant drift detected. Review the shifted telemetry features below.")
        drift_cols[1].metric("Largest shifted feature", drift_values["Largest shifted feature"])
    drift_cols[2].metric("Average drift", f"{drift_values['Average drift']:.3f}")
    st.write("##### Shifted features")
    if shifted_features.empty:
        st.caption("No features exceeded the configured drift threshold.")
    else:
        st.dataframe(style_feature_drift_dataframe(shifted_features), use_container_width=True)
    st.write("##### Label drift")
    st.dataframe(style_label_drift_dataframe(label_drift_df), use_container_width=True)


def _render_anomaly_detection(st: Any, anomaly_report: Optional[AnomalyReport]) -> None:
    st.subheader("Anomaly Detection")
    st.caption("Identifies unusual interview sessions using unsupervised learning.")
    if anomaly_report is None:
        st.info("Run ML Analytics to generate anomaly detection analytics.")
        return

    anomaly_summary, cluster_df = build_anomaly_summary(anomaly_report)
    anomaly_cols = st.columns(4)
    anomaly_cols[0].metric("Anomalies", anomaly_summary["Anomalies"])
    anomaly_cols[1].metric("Noise points", anomaly_summary["Noise points"])
    anomaly_cols[2].metric("Unknown failures", anomaly_summary["Unknown failure candidates"])
    anomaly_cols[3].metric("Cluster count", anomaly_summary["Cluster count"])
    if int(anomaly_summary["Cluster count"]) == 0:
        st.info(
            "No meaningful clusters were identified in the uploaded dataset.\n\n"
            "This is expected when the uploaded data closely matches the baseline reference."
        )
    elif int(anomaly_summary["Cluster count"]) == 1:
        st.info(
            "The uploaded dataset does not contain sufficient separable structure for "
            "meaningful clustering. This is expected when analysing the baseline dataset."
        )
    else:
        st.write("##### DBSCAN cluster statistics")
        st.dataframe(cluster_df, use_container_width=True)
    records = anomaly_report.get("records")
    if isinstance(records, list) and records:
        with st.expander("Show anomaly records"):
            st.dataframe(
                pd.DataFrame(records)[
                    ["session_id", "anomaly_score", "is_anomaly", "cluster_id", "is_noise", "is_unknown_failure"]
                ],
                use_container_width=True,
            )


def _render_export_buttons(
    st: Any,
    evaluation_report: Optional[EvaluationReport],
    importance_reports: List[FeatureImportanceReport],
    drift_report: Optional[DriftReport],
    anomaly_report: Optional[AnomalyReport],
) -> None:
    with _card_container(st):
        st.subheader("Reports")
        st.caption(
            "Download the unchanged, serializable outputs from each offline "
            "analytics stage."
        )
        eval_payload = create_download_payload(evaluation_report)
        importance_payload = create_download_payload(importance_reports[0] if importance_reports else None)
        drift_payload = create_download_payload(drift_report)
        anomaly_payload = create_download_payload(anomaly_report)
        export_cols = st.columns(4)
        export_cols[0].download_button(
            "Download evaluation report", data=eval_payload or "{}", file_name="evaluation_report.json",
            mime="application/json", disabled=eval_payload is None,
        )
        export_cols[1].download_button(
            "Download feature importance report", data=importance_payload or "{}", file_name="feature_importance_report.json",
            mime="application/json", disabled=importance_payload is None,
        )
        export_cols[2].download_button(
            "Download drift report", data=drift_payload or "{}", file_name="drift_report.json",
            mime="application/json", disabled=drift_payload is None,
        )
        export_cols[3].download_button(
            "Download anomaly report", data=anomaly_payload or "{}", file_name="anomaly_report.json",
            mime="application/json", disabled=anomaly_payload is None,
        )


def render_ml_analytics() -> None:
    """Render the ML analytics tab for the Streamlit dashboard."""
    import streamlit as st

    st.header("📊 ML Analytics")
    st.write(
        "Upload a dataset file and run the ML analytics workflow. The dashboard will "
        "execute evaluation, feature importance, drift detection, and anomaly analysis automatically."
    )

    if "ml_dataset_key" not in st.session_state:
        st.session_state["ml_dataset_key"] = None
    if "ml_analytics_results" not in st.session_state:
        st.session_state["ml_analytics_results"] = None

    with st.expander("Upload dataset and run analytics", expanded=True):
        dataset_col, action_col = st.columns([3, 1])
        dataset_file = dataset_col.file_uploader(
            "Dataset file",
            type=["jsonl", "ndjson", "csv"],
            key="ml_dataset_upload",
            help="Upload a synthetic interview sessions dataset for ML analytics.",
        )
        run_analytics = action_col.button("Run ML Analytics", type="primary")

    dataset_statistics: Optional[DatasetStatistics] = None
    dataset: Optional[MLDataset] = None
    dataset_error: Optional[str] = None
    dataset_key = DEFAULT_DATASET_KEY
    dataset_bytes: Optional[bytes] = None
    dataset_suffix = ".jsonl"

    if dataset_file is not None:
        dataset_bytes = dataset_file.getvalue()
        dataset_suffix = Path(dataset_file.name).suffix or ".jsonl"
        dataset_key = dataset_cache_key(dataset_file.name, dataset_bytes)

    try:
        if dataset_bytes is not None:
            dataset_statistics, dataset = load_dataset_from_bytes(dataset_bytes, dataset_suffix)
        else:
            dataset_statistics, dataset = load_dataset_statistics(DEFAULT_DATASET_PATH)
    except Exception as error:
        dataset_error = str(error)

    if run_analytics and dataset is not None and dataset_error is None:
        if st.session_state["ml_dataset_key"] != dataset_key:
            st.session_state["ml_analytics_results"] = None
        try:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            def status_update(percent: int, message: str) -> None:
                progress_bar.progress(percent)
                status_placeholder.info(message)

            _, baseline_dataset = load_dataset_statistics(DEFAULT_DATASET_PATH)
            ml_results = run_ml_analytics(
                dataset,
                baseline_dataset,
                random_state=42,
                progress_callback=status_update,
            )
            st.session_state["ml_dataset_key"] = dataset_key
            st.session_state["ml_dataset_statistics_cache"] = dataset_statistics
            st.session_state["ml_dataset_cache"] = dataset
            st.session_state["ml_analytics_results"] = ml_results
        except Exception as error:
            st.error(f"ML analytics failed: {error}")
            st.session_state["ml_analytics_results"] = None

    if st.session_state["ml_dataset_key"] == dataset_key and st.session_state["ml_analytics_results"] is not None:
        ml_results = st.session_state["ml_analytics_results"]
    else:
        ml_results = None

    _render_analytics_summary(
        st,
        dataset_statistics,
        dataset,
        ml_results["evaluation_report"] if ml_results else None,
        ml_results["drift_report"] if ml_results else None,
        ml_results["anomaly_report"] if ml_results else None,
    )
    st.markdown("---")
    _render_dataset_overview(
        st,
        dataset_statistics,
        dataset,
        dataset_error,
        ml_results["evaluation_report"] if ml_results else None,
    )
    st.markdown("---")
    _render_model_evaluation(
        st,
        ml_results["evaluation_report"] if ml_results else None,
        ml_results["feature_importance_reports"] if ml_results else [],
    )
    st.markdown("---")
    _render_feature_importance(st, ml_results["feature_importance_reports"] if ml_results else [])
    st.markdown("---")
    _render_drift_detection(st, ml_results["drift_report"] if ml_results else None)
    st.markdown("---")
    _render_anomaly_detection(st, ml_results["anomaly_report"] if ml_results else None)
    st.markdown("---")
    _render_export_buttons(
        st,
        ml_results["evaluation_report"] if ml_results else None,
        ml_results["feature_importance_reports"] if ml_results else [],
        ml_results["drift_report"] if ml_results else None,
        ml_results["anomaly_report"] if ml_results else None,
    )
