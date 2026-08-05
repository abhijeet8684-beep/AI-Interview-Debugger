"""Offline supervised model evaluation for stable extracted-signal features."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.dataset_loader import MLDataset

LabelKind = Literal["root_cause", "affected_component", "failure_category"]


@dataclass(frozen=True)
class ModelEvaluation:
    """Serializable evaluation outcome for one trained classifier."""

    model_name: str
    metrics: Dict[str, float]
    confusion_matrix: Tuple[Tuple[int, ...], ...]
    class_labels: Tuple[str, ...]
    classification_report: Dict[str, Dict[str, float]]
    train_size: int
    test_size: int

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation of this evaluation."""
        return asdict(self)


@dataclass(frozen=True)
class BestModelMetadata:
    """Minimal persisted metadata for the highest-ranked offline model."""

    model_name: str
    macro_f1: float
    feature_names: Tuple[str, ...]
    label_kind: str

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-serializable best-model metadata only."""
        return {
            "model_name": self.model_name,
            "macro_f1": self.macro_f1,
            "feature_names": list(self.feature_names),
            "label_kind": self.label_kind,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Complete model-comparison result for one offline evaluation run."""

    evaluations: Tuple[ModelEvaluation, ...]
    ranking: Tuple[str, ...]
    best_model: BestModelMetadata

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable evaluation report."""
        return {
            "evaluations": [item.to_dict() for item in self.evaluations],
            "ranking": list(self.ranking),
            "best_model": self.best_model.to_dict(),
        }


class ModelEvaluationError(ValueError):
    """Raised when a dataset cannot support deterministic supervised evaluation."""


class ModelEvaluator:
    """Train and compare offline sklearn classifiers on ``MLDataset`` features.

    Models are used only for offline analytics. They are never connected to the
    deterministic diagnosis, evidence, or confidence pipeline.
    """

    def __init__(self, random_state: int = 42, test_size: float = 0.25) -> None:
        """Initialize reproducible evaluation configuration.

        Args:
            random_state: Seed passed to stochastic sklearn estimators and split.
            test_size: Fraction reserved for deterministic holdout evaluation.
        """
        self._random_state = random_state
        self._test_size = test_size

    def evaluate(
        self,
        dataset: MLDataset,
        label_kind: LabelKind = "failure_category",
        metadata_path: Path | None = None,
    ) -> EvaluationReport:
        """Train and compare all supported models on one labeled dataset.

        Args:
            dataset: Stable feature matrix and aligned ground-truth labels.
            label_kind: Ground-truth target to evaluate.
            metadata_path: Optional JSON destination for best-model metadata.

        Returns:
            Structured model metrics, ranking, and best-model metadata.

        Raises:
            ModelEvaluationError: If features or labels cannot support a split.
        """
        features, labels = self._validated_data(dataset, label_kind)
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=labels,
        )
        class_labels = tuple(sorted(set(labels)))
        evaluations = tuple(
            self._evaluate_model(name, model, train_x, test_x, train_y, test_y, class_labels)
            for name, model in self._models()
        )
        ordered = tuple(
            sorted(evaluations, key=lambda item: (-item.metrics["macro_f1"], item.model_name))
        )
        best = ordered[0]
        report = EvaluationReport(
            evaluations=ordered,
            ranking=tuple(item.model_name for item in ordered),
            best_model=BestModelMetadata(
                model_name=best.model_name,
                macro_f1=best.metrics["macro_f1"],
                feature_names=dataset.feature_names,
                label_kind=label_kind,
            ),
        )
        if metadata_path is not None:
            self.persist_best_metadata(report, metadata_path)
        return report

    def persist_best_metadata(self, report: EvaluationReport, path: Path) -> None:
        """Persist only best-model metadata as deterministic JSON.

        Args:
            report: Completed offline evaluation report.
            path: Output JSON path for best-model metadata.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.best_model.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _validated_data(
        self, dataset: MLDataset, label_kind: LabelKind
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate aligned, fully labeled data for a stratified model split."""
        labels = self._labels(dataset, label_kind)
        if dataset.features.ndim != 2 or dataset.features.shape[0] == 0:
            raise ModelEvaluationError("Dataset must contain at least one feature row.")
        if len(labels) != dataset.features.shape[0] or any(label is None for label in labels):
            raise ModelEvaluationError("Selected labels must be present for every feature row.")
        string_labels = np.asarray(labels, dtype=str)
        counts = {label: int((string_labels == label).sum()) for label in set(string_labels)}
        if len(counts) < 2:
            raise ModelEvaluationError("At least two label classes are required.")
        if min(counts.values()) < 2:
            raise ModelEvaluationError("Each label class requires at least two samples.")
        if not 0.0 < self._test_size < 1.0:
            raise ModelEvaluationError("test_size must be between zero and one.")
        return dataset.features, string_labels

    def _labels(self, dataset: MLDataset, label_kind: LabelKind) -> Sequence[str | None]:
        """Return the requested ground-truth label sequence from a dataset."""
        choices = {
            "root_cause": dataset.labels.root_causes,
            "affected_component": dataset.labels.affected_components,
            "failure_category": dataset.labels.failure_categories,
        }
        try:
            return choices[label_kind]
        except KeyError as error:
            raise ModelEvaluationError(f"Unsupported label kind: {label_kind}") from error

    def _models(self) -> Tuple[Tuple[str, Pipeline], ...]:
        """Return reproducible supported sklearn model pipelines."""
        return (
            (
                "Logistic Regression",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1_000, random_state=self._random_state)),
                ]),
            ),
            (
                "Random Forest",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=100, random_state=self._random_state)),
                ]),
            ),
            (
                "Gradient Boosting",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingClassifier(random_state=self._random_state)),
                ]),
            ),
        )

    def _evaluate_model(
        self,
        name: str,
        model: Pipeline,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
        class_labels: Tuple[str, ...],
    ) -> ModelEvaluation:
        """Fit one model and calculate consistent holdout metrics."""
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        report = classification_report(
            test_y, predictions, labels=class_labels, output_dict=True, zero_division=0
        )
        metrics = {
            "accuracy": float(accuracy_score(test_y, predictions)),
            "precision": float(precision_score(test_y, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(test_y, predictions, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(test_y, predictions, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(test_y, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(test_y, predictions, average="weighted", zero_division=0)),
            "micro_f1": float(f1_score(test_y, predictions, average="micro", zero_division=0)),
        }
        return ModelEvaluation(
            model_name=name,
            metrics=metrics,
            confusion_matrix=tuple(
                tuple(int(value) for value in row)
                for row in confusion_matrix(test_y, predictions, labels=class_labels)
            ),
            class_labels=class_labels,
            classification_report=self._serialize_report(report),
            train_size=len(train_y),
            test_size=len(test_y),
        )

    def _serialize_report(
        self, report: Dict[str, object]
    ) -> Dict[str, Dict[str, float]]:
        """Convert sklearn report scalars and mappings into JSON-safe floats."""
        serialized: Dict[str, Dict[str, float]] = {}
        for name, values in report.items():
            if isinstance(values, dict):
                serialized[name] = {key: float(value) for key, value in values.items()}
            else:
                serialized[name] = {"score": float(values)}
        return serialized
