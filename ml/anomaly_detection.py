"""Offline deterministic anomaly and unknown-failure discovery for ML data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ml.dataset_loader import MLDataset


@dataclass(frozen=True)
class AnomalyRecord:
    """Offline anomaly and cluster assignment for one historical session."""

    session_id: str
    anomaly_score: float
    is_anomaly: bool
    cluster_id: int
    is_noise: bool
    is_unknown_failure: bool


@dataclass(frozen=True)
class ClusterStatistic:
    """Count summary for one DBSCAN cluster or noise label."""

    cluster_id: int
    session_count: int
    is_noise: bool


@dataclass(frozen=True)
class AnomalyReport:
    """Immutable, JSON-serializable offline anomaly-discovery output."""

    records: Tuple[AnomalyRecord, ...]
    cluster_statistics: Tuple[ClusterStatistic, ...]
    anomaly_count: int
    noise_point_count: int
    unknown_failure_count: int

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-native anomaly report data."""
        return {
            "records": [item.__dict__ for item in self.records],
            "cluster_statistics": [item.__dict__ for item in self.cluster_statistics],
            "anomaly_count": self.anomaly_count,
            "noise_point_count": self.noise_point_count,
            "unknown_failure_count": self.unknown_failure_count,
        }

    def to_json(self) -> str:
        """Return deterministic JSON text for offline report storage."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class AnomalyDetectionError(ValueError):
    """Raised when an ML feature matrix cannot support anomaly detection."""


class AnomalyDetector:
    """Discover outliers and clusters with deterministic sklearn algorithms.

    This class is offline analytics only; results never affect deterministic
    diagnosis, confidence, evidence, or runtime pipeline behavior.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        eps: float = 0.8,
        min_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize deterministic Isolation Forest and DBSCAN parameters."""
        self._contamination = contamination
        self._eps = eps
        self._min_samples = min_samples
        self._random_state = random_state

    def analyze(self, dataset: MLDataset) -> AnomalyReport:
        """Calculate outlier scores, DBSCAN clusters, and unknown-failure flags."""
        self._validate(dataset)
        if not len(dataset.session_ids):
            return AnomalyReport((), (), 0, 0, 0)
        features = self._prepared_features(dataset.features)
        forest = IsolationForest(
            contamination=self._contamination,
            random_state=self._random_state,
        ).fit(features)
        anomaly_scores = -forest.score_samples(features)
        anomaly_flags = forest.predict(features) == -1
        cluster_ids = DBSCAN(eps=self._eps, min_samples=self._min_samples).fit_predict(features)
        cluster_counts = self._cluster_counts(cluster_ids)
        records = tuple(
            AnomalyRecord(
                session_id=session_id,
                anomaly_score=float(score),
                is_anomaly=bool(is_anomaly),
                cluster_id=int(cluster_id),
                is_noise=bool(cluster_id == -1),
                is_unknown_failure=bool(is_anomaly or cluster_id == -1 or cluster_counts[int(cluster_id)] <= 1),
            )
            for session_id, score, is_anomaly, cluster_id in zip(
                dataset.session_ids, anomaly_scores, anomaly_flags, cluster_ids
            )
        )
        statistics = tuple(
            ClusterStatistic(cluster_id=cluster_id, session_count=count, is_noise=cluster_id == -1)
            for cluster_id, count in sorted(cluster_counts.items())
        )
        return AnomalyReport(
            records=records,
            cluster_statistics=statistics,
            anomaly_count=sum(record.is_anomaly for record in records),
            noise_point_count=sum(record.is_noise for record in records),
            unknown_failure_count=sum(record.is_unknown_failure for record in records),
        )

    def export_json(self, report: AnomalyReport, path: Path) -> None:
        """Persist a deterministic offline anomaly report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.to_json(), encoding="utf-8")

    def _validate(self, dataset: MLDataset) -> None:
        """Validate feature metadata and Isolation Forest input constraints."""
        if dataset.features.ndim != 2:
            raise AnomalyDetectionError("Dataset features must be two-dimensional.")
        if dataset.features.shape[1] != len(dataset.feature_names):
            raise AnomalyDetectionError("Feature dimensions do not match metadata.")
        if dataset.features.shape[0] != len(dataset.session_ids):
            raise AnomalyDetectionError("Feature rows must align with session identifiers.")
        if not 0.0 < self._contamination <= 0.5:
            raise AnomalyDetectionError("contamination must be in the interval (0, 0.5].")

    def _prepared_features(self, features: np.ndarray) -> np.ndarray:
        """Median-impute and standardize features for both offline algorithms."""
        return StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(features))

    def _cluster_counts(self, cluster_ids: np.ndarray) -> Dict[int, int]:
        """Return deterministic DBSCAN cluster-size lookup."""
        counts: Dict[int, int] = {}
        for cluster_id in cluster_ids:
            identifier = int(cluster_id)
            counts[identifier] = counts.get(identifier, 0) + 1
        return counts
