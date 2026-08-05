"""Deterministic sklearn-compatible feature-importance analysis utilities."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

ImportanceMethod = Literal["auto", "native", "coefficient", "permutation"]


@dataclass(frozen=True)
class FeatureImportance:
    """Importance of one stable feature in a fitted offline model."""

    feature_name: str
    importance_score: float
    normalized_score: float
    rank: int

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-native importance data."""
        return asdict(self)


@dataclass(frozen=True)
class ImportanceSummary:
    """Descriptive summary of a deterministic feature-importance ranking."""

    top_features: Tuple[FeatureImportance, ...]
    bottom_features: Tuple[FeatureImportance, ...]
    mean_importance: float
    median_importance: float
    standard_deviation: float

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-native summary data."""
        return {
            "top_features": [item.to_dict() for item in self.top_features],
            "bottom_features": [item.to_dict() for item in self.bottom_features],
            "mean_importance": self.mean_importance,
            "median_importance": self.median_importance,
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True)
class FeatureImportanceReport:
    """Immutable explainability output for one fitted estimator."""

    model_name: str
    ranking: Tuple[FeatureImportance, ...]
    summary: ImportanceSummary
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        """Return a complete JSON-native explainability report."""
        return {
            "model_name": self.model_name,
            "ranking": [item.to_dict() for item in self.ranking],
            "summary": self.summary.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Return deterministic JSON text for report interchange."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class FeatureImportanceError(ValueError):
    """Raised when an estimator cannot provide deterministic importance data."""


class FeatureImportanceAnalyzer:
    """Extract native, coefficient, or permutation feature importance.

    The analyzer is model-agnostic and used only for offline ML analytics. It
    does not interact with deterministic diagnosis or pipeline components.
    """

    def __init__(self, random_state: int = 42, n_repeats: int = 10) -> None:
        """Initialize deterministic permutation-importance configuration.

        Args:
            random_state: Fixed seed passed to sklearn permutation importance.
            n_repeats: Number of deterministic permutation repeats.
        """
        self._random_state = random_state
        self._n_repeats = n_repeats

    def analyze(
        self,
        model: object,
        feature_names: Sequence[str],
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        method: ImportanceMethod = "auto",
    ) -> FeatureImportanceReport:
        """Calculate a ranked deterministic importance report for a fitted model.

        Args:
            model: Fitted sklearn estimator or fitted sklearn pipeline.
            feature_names: Stable feature names aligned with model input columns.
            features: Evaluation features, required for permutation importance.
            labels: Evaluation labels, required for permutation importance.
            method: Importance strategy or automatic compatible-method selection.

        Returns:
            Immutable ranked feature-importance report.

        Raises:
            FeatureImportanceError: If the estimator is unsupported, unfitted, or
                supplied features do not match the stable feature metadata.
        """
        names = tuple(feature_names)
        estimator = self._final_estimator(model)
        self._validate_feature_names(estimator, names)
        selected_method = self._method(estimator, method)
        scores = self._scores(model, estimator, features, labels, selected_method)
        if len(scores) != len(names):
            raise FeatureImportanceError("Importance scores do not match feature names.")
        ranking = self._ranking(names, scores)
        return FeatureImportanceReport(
            model_name=estimator.__class__.__name__,
            ranking=ranking,
            summary=self._summary(ranking),
            metadata={
                "method": selected_method,
                "feature_count": len(names),
                "random_state": self._random_state if selected_method == "permutation" else None,
            },
        )

    def export_json(self, report: FeatureImportanceReport, path: Path) -> None:
        """Persist a deterministic JSON feature-importance report.

        Args:
            report: Completed immutable importance report.
            path: Destination JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.to_json(), encoding="utf-8")

    def _final_estimator(self, model: object) -> object:
        """Return the fitted final estimator while retaining the original model."""
        if isinstance(model, Pipeline):
            if not model.steps:
                raise FeatureImportanceError("Pipeline does not contain an estimator.")
            return model.steps[-1][1]
        return model

    def _validate_feature_names(self, estimator: object, names: Tuple[str, ...]) -> None:
        """Validate non-empty stable metadata against fitted estimator dimensions."""
        if not names:
            raise FeatureImportanceError("At least one feature name is required.")
        expected_count = getattr(estimator, "n_features_in_", None)
        if expected_count is not None and expected_count != len(names):
            raise FeatureImportanceError("Feature name count does not match fitted estimator input.")

    def _method(self, estimator: object, method: ImportanceMethod) -> str:
        """Resolve an explicit or automatically compatible importance method."""
        compatible = {
            "native": hasattr(estimator, "feature_importances_"),
            "coefficient": hasattr(estimator, "coef_"),
        }
        if method == "auto":
            if compatible["native"]:
                return "native"
            if compatible["coefficient"]:
                return "coefficient"
            raise FeatureImportanceError("Estimator exposes no native importance interface.")
        if method == "permutation":
            return method
        if not compatible.get(method, False):
            raise FeatureImportanceError(f"Estimator does not support {method} importance.")
        return method

    def _scores(
        self,
        model: object,
        estimator: object,
        features: np.ndarray | None,
        labels: np.ndarray | None,
        method: str,
    ) -> np.ndarray:
        """Return non-negative raw importance scores for the selected method."""
        if method == "native":
            return np.abs(np.asarray(getattr(estimator, "feature_importances_"), dtype=float))
        if method == "coefficient":
            coefficients = np.asarray(getattr(estimator, "coef_"), dtype=float)
            return np.mean(np.abs(np.atleast_2d(coefficients)), axis=0)
        if features is None or labels is None:
            raise FeatureImportanceError("Permutation importance requires features and labels.")
        try:
            result = permutation_importance(
                model,
                features,
                labels,
                n_repeats=self._n_repeats,
                random_state=self._random_state,
                scoring="f1_weighted",
            )
        except Exception as error:
            raise FeatureImportanceError("Permutation importance could not be calculated.") from error
        return np.abs(np.asarray(result.importances_mean, dtype=float))

    def _ranking(
        self, names: Tuple[str, ...], scores: np.ndarray
    ) -> Tuple[FeatureImportance, ...]:
        """Normalize and rank scores with feature-name tie-breaking."""
        total = float(scores.sum())
        normalized = scores / total if total > 0.0 else np.zeros_like(scores)
        ordered = sorted(zip(names, scores, normalized), key=lambda item: (-item[1], item[0]))
        return tuple(
            FeatureImportance(
                feature_name=name,
                importance_score=float(score),
                normalized_score=float(normalized_score),
                rank=index,
            )
            for index, (name, score, normalized_score) in enumerate(ordered, start=1)
        )

    def _summary(self, ranking: Tuple[FeatureImportance, ...]) -> ImportanceSummary:
        """Calculate top/bottom ten and descriptive score statistics."""
        scores = np.asarray([item.importance_score for item in ranking], dtype=float)
        bottom = tuple(sorted(ranking, key=lambda item: (item.importance_score, item.feature_name))[:10])
        return ImportanceSummary(
            top_features=ranking[:10],
            bottom_features=bottom,
            mean_importance=float(scores.mean()),
            median_importance=float(np.median(scores)),
            standard_deviation=float(scores.std()),
        )
