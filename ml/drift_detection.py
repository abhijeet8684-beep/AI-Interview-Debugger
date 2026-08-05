"""Offline deterministic feature and label drift analysis for ML datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ml.dataset_loader import MLDataset


@dataclass(frozen=True)
class FeatureDrift:
    """Drift measurements for one stable extracted-signal feature."""

    feature_name: str
    jensen_shannon_divergence: float
    population_stability_index: float
    shifted: bool


@dataclass(frozen=True)
class LabelDrift:
    """Percentage-point distribution change for one label value."""

    label_group: str
    label: str
    baseline_percentage: float
    current_percentage: float
    percentage_change: float


@dataclass(frozen=True)
class DriftSummary:
    """Aggregate deterministic drift measures across a dataset comparison."""

    shifted_feature_count: int
    largest_drift: float
    average_drift: float
    overall_drift_score: float


@dataclass(frozen=True)
class DriftReport:
    """Immutable, JSON-serializable comparison of baseline and current datasets."""

    feature_drift: Tuple[FeatureDrift, ...]
    label_drift: Tuple[LabelDrift, ...]
    summary: DriftSummary

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-native drift data in stable feature and label order."""
        return {
            "feature_drift": [item.__dict__ for item in self.feature_drift],
            "label_drift": [item.__dict__ for item in self.label_drift],
            "summary": self.summary.__dict__,
        }

    def to_json(self) -> str:
        """Return deterministic JSON text for offline report storage."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class DriftDetectionError(ValueError):
    """Raised when datasets cannot be compared through stable feature metadata."""


class DriftDetector:
    """Compare historical datasets with deterministic JS divergence and PSI.

    This class is offline analytics only. It does not influence deterministic
    diagnosis, confidence, evidence, or production runtime behavior.
    """

    def __init__(self, shift_threshold: float = 0.1, bins: int = 10) -> None:
        """Initialize deterministic drift thresholds and histogram resolution."""
        self._shift_threshold = shift_threshold
        self._bins = bins

    def compare(self, baseline: MLDataset, current: MLDataset) -> DriftReport:
        """Compare stable feature and ground-truth label distributions.

        Args:
            baseline: Reference historical ML dataset.
            current: Current historical ML dataset.

        Returns:
            Immutable feature, label, and aggregate drift report.

        Raises:
            DriftDetectionError: If stable feature metadata is incompatible.
        """
        self._validate_datasets(baseline, current)
        feature_drift = tuple(
            self._feature_drift(name, baseline.features[:, index], current.features[:, index])
            for index, name in enumerate(baseline.feature_names)
        )
        label_drift = tuple(
            item
            for group, baseline_labels, current_labels in self._label_groups(baseline, current)
            for item in self._label_drift(group, baseline_labels, current_labels)
        )
        scores = [item.jensen_shannon_divergence for item in feature_drift]
        summary = DriftSummary(
            shifted_feature_count=sum(item.shifted for item in feature_drift),
            largest_drift=max(scores, default=0.0),
            average_drift=float(np.mean(scores)) if scores else 0.0,
            overall_drift_score=float(np.mean(scores)) if scores else 0.0,
        )
        return DriftReport(feature_drift, label_drift, summary)

    def export_json(self, report: DriftReport, path: Path) -> None:
        """Persist a deterministic offline drift report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.to_json(), encoding="utf-8")

    def _validate_datasets(self, baseline: MLDataset, current: MLDataset) -> None:
        """Validate feature dimensions and immutable ordering before comparison."""
        if baseline.feature_names != current.feature_names:
            raise DriftDetectionError("Datasets must use identical feature ordering.")
        expected = len(baseline.feature_names)
        if baseline.features.ndim != 2 or current.features.ndim != 2:
            raise DriftDetectionError("Dataset features must be two-dimensional.")
        if baseline.features.shape[1] != expected or current.features.shape[1] != expected:
            raise DriftDetectionError("Dataset feature dimensions do not match metadata.")

    def _feature_drift(
        self, name: str, baseline: np.ndarray, current: np.ndarray
    ) -> FeatureDrift:
        """Calculate JS divergence and PSI for one numeric feature."""
        baseline_values = baseline[~np.isnan(baseline)]
        current_values = current[~np.isnan(current)]
        if not len(baseline_values) and not len(current_values):
            return FeatureDrift(name, 0.0, 0.0, False)
        if not len(baseline_values) or not len(current_values):
            return FeatureDrift(name, 1.0, 1.0, True)
        minimum = min(float(baseline_values.min()), float(current_values.min()))
        maximum = max(float(baseline_values.max()), float(current_values.max()))
        if minimum == maximum:
            return FeatureDrift(name, 0.0, 0.0, False)
        edges = np.linspace(minimum, maximum, self._bins + 1)
        baseline_distribution = self._distribution(baseline_values, edges)
        current_distribution = self._distribution(current_values, edges)
        midpoint = 0.5 * (baseline_distribution + current_distribution)
        js_divergence = 0.5 * (
            self._kl_divergence(baseline_distribution, midpoint)
            + self._kl_divergence(current_distribution, midpoint)
        )
        psi = float(np.sum(
            (current_distribution - baseline_distribution)
            * np.log(current_distribution / baseline_distribution)
        ))
        return FeatureDrift(
            name,
            float(js_divergence),
            psi,
            float(js_divergence) >= self._shift_threshold,
        )

    def _distribution(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Return epsilon-smoothed normalized histogram probabilities."""
        counts, _ = np.histogram(values, bins=edges)
        probabilities = counts.astype(float) + 1e-12
        return probabilities / probabilities.sum()

    def _kl_divergence(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return a finite KL divergence for smoothed distributions."""
        return float(np.sum(left * np.log(left / right)))

    def _label_groups(
        self, baseline: MLDataset, current: MLDataset
    ) -> Tuple[Tuple[str, Sequence[str | None], Sequence[str | None]], ...]:
        """Return aligned named label sequences for distribution comparison."""
        return (
            ("root_cause", baseline.labels.root_causes, current.labels.root_causes),
            ("failure_category", baseline.labels.failure_categories, current.labels.failure_categories),
            ("affected_component", baseline.labels.affected_components, current.labels.affected_components),
        )

    def _label_drift(
        self, group: str, baseline: Sequence[str | None], current: Sequence[str | None]
    ) -> List[LabelDrift]:
        """Calculate stable percentage-point changes for one label group."""
        baseline_distribution = self._label_distribution(baseline)
        current_distribution = self._label_distribution(current)
        return [
            LabelDrift(
                group,
                label,
                baseline_distribution.get(label, 0.0),
                current_distribution.get(label, 0.0),
                current_distribution.get(label, 0.0) - baseline_distribution.get(label, 0.0),
            )
            for label in sorted(set(baseline_distribution) | set(current_distribution))
        ]

    def _label_distribution(self, labels: Sequence[str | None]) -> Dict[str, float]:
        """Return percentage distributions, preserving unavailable labels explicitly."""
        if not labels:
            return {}
        counts: Dict[str, int] = {}
        for label in labels:
            key = label or "unlabeled"
            counts[key] = counts.get(key, 0) + 1
        return {label: 100.0 * count / len(labels) for label, count in counts.items()}
