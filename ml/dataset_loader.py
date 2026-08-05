"""Schema-safe dataset loading and stable signal feature engineering for ML.

JSONL is the authoritative format because it preserves complete
``InterviewSession`` records. CSV support is intentionally limited to the
signal-compatible fields exported by the synthetic dataset generator.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import ValidationError

from pipeline.schema import ExtractedSignal, InterviewSession
from pipeline.signal_extractor import SignalExtractor


@dataclass(frozen=True)
class LoadIssue:
    """A non-fatal malformed source row encountered during dataset loading."""

    row_number: int
    message: str


@dataclass(frozen=True)
class LabelSet:
    """Ground-truth labels aligned with feature-matrix rows.

    ``None`` represents an intentionally unavailable label, including successful
    sessions that have no ground-truth root cause.
    """

    root_causes: Tuple[Optional[str], ...]
    affected_components: Tuple[Optional[str], ...]
    failure_categories: Tuple[Optional[str], ...]


@dataclass(frozen=True)
class MLDataset:
    """Deterministic ML-ready data produced from generated session records."""

    features: np.ndarray
    feature_names: Tuple[str, ...]
    labels: LabelSet
    session_ids: Tuple[str, ...]
    source_format: str
    issues: Tuple[LoadIssue, ...]


@dataclass(frozen=True)
class DatasetStatistics:
    """Summary metrics for ML dataset completeness and label distribution."""

    feature_count: int
    row_count: int
    class_balance: Dict[str, int]
    missing_values: Dict[str, int]
    summary: str


class DatasetValidationError(ValueError):
    """Raised when strict dataset loading encounters an invalid source row."""


class DatasetLoader:
    """Load generated datasets and reuse frozen signal engineering for ML.

    The stable feature order is deliberately explicit because model artifacts
    from future phases require a fixed column contract.
    """

    _FEATURE_NAMES: Tuple[str, ...] = (
        "interview_duration_seconds",
        "transcript_turn_count",
        "interviewer_turn_count",
        "candidate_turn_count",
        "interview_stage_count",
        "total_llm_calls",
        "failed_llm_calls",
        "average_llm_latency_ms",
        "maximum_llm_latency_ms",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "average_response_time",
        "total_tool_calls",
        "failed_tool_calls",
        "average_tool_latency",
        "maximum_tool_latency",
        "retry_count",
        "total_events",
        "timeout_events",
        "retry_events",
        "failure_events",
        "average_candidate_response_length",
        "average_interviewer_question_length",
        "empty_candidate_responses",
        "empty_transcript_turns",
        "high_latency_detected",
        "multiple_retries",
        "empty_transcript",
        "json_errors_detected",
        "tool_failures_detected",
    )

    def __init__(self, signal_extractor: Optional[SignalExtractor] = None) -> None:
        """Initialize with an injectable frozen signal extractor.

        Args:
            signal_extractor: Extractor reused for JSONL feature construction.
        """
        self._signal_extractor = signal_extractor or SignalExtractor()

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Return the immutable, stable feature ordering contract."""
        return self._FEATURE_NAMES

    def load(self, path: Path, strict: bool = False) -> MLDataset:
        """Load JSONL or CSV data based on a source file extension.

        Args:
            path: JSONL or CSV source path.
            strict: Raise on malformed rows instead of collecting issues.

        Returns:
            ML-ready feature matrix, aligned labels, and source metadata.

        Raises:
            ValueError: If the extension is unsupported.
        """
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            sessions, issues = self.load_jsonl(path, strict=strict)
            return self.build_from_sessions(sessions, issues)
        if suffix == ".csv":
            rows, issues = self.load_csv(path, strict=strict)
            return self.build_from_csv_rows(rows, issues)
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    def load_jsonl(
        self, path: Path, strict: bool = False
    ) -> Tuple[List[InterviewSession], Tuple[LoadIssue, ...]]:
        """Load full session records from JSONL in deterministic file order."""
        sessions: List[InterviewSession] = []
        issues: List[LoadIssue] = []
        with path.open("r", encoding="utf-8") as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    sessions.append(InterviewSession.model_validate_json(raw_line))
                except ValidationError as error:
                    self._record_issue(issues, row_number, str(error), strict)
        return sessions, tuple(issues)

    def load_csv(
        self, path: Path, strict: bool = False
    ) -> Tuple[List[Dict[str, str]], Tuple[LoadIssue, ...]]:
        """Load flattened generator CSV rows in deterministic file order.

        CSV rows are not reconstructed as sessions because the flattened format
        omits transcript and call-level schema data.
        """
        rows: List[Dict[str, str]] = []
        issues: List[LoadIssue] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "session_id" not in reader.fieldnames:
                raise DatasetValidationError("CSV must contain a session_id column.")
            for row_number, row in enumerate(reader, start=2):
                if not row.get("session_id"):
                    self._record_issue(issues, row_number, "Missing session_id.", strict)
                    continue
                rows.append({key: value or "" for key, value in row.items()})
        return rows, tuple(issues)

    def build_from_sessions(
        self,
        sessions: Sequence[InterviewSession],
        issues: Sequence[LoadIssue] = (),
    ) -> MLDataset:
        """Build a complete stable feature matrix from schema-valid sessions."""
        feature_rows = [self._signal_row(self._session_signals(session)) for session in sessions]
        labels = self._labels_from_sessions(sessions)
        return self._dataset(
            feature_rows, labels, [str(session.session_id) for session in sessions],
            "jsonl", issues,
        )

    def build_from_csv_rows(
        self,
        rows: Sequence[Dict[str, str]],
        issues: Sequence[LoadIssue] = (),
    ) -> MLDataset:
        """Build a partial signal matrix from flattened generator CSV rows."""
        feature_rows = [self._csv_signal_row(row) for row in rows]
        labels = LabelSet(
            root_causes=tuple(self._optional_value(row.get("ground_truth_failure_type")) for row in rows),
            affected_components=tuple(self._optional_value(row.get("ground_truth_component")) for row in rows),
            failure_categories=tuple(self._optional_value(row.get("ground_truth_scenario")) for row in rows),
        )
        return self._dataset(feature_rows, labels, [row["session_id"] for row in rows], "csv", issues)

    def statistics(self, dataset: MLDataset) -> DatasetStatistics:
        """Return feature completeness and ground-truth root-cause distribution."""
        missing_values = {
            name: int(np.isnan(dataset.features[:, index]).sum()) if dataset.features.size else 0
            for index, name in enumerate(dataset.feature_names)
        }
        class_balance: Dict[str, int] = {}
        for label in dataset.labels.root_causes:
            key = label or "unlabeled"
            class_balance[key] = class_balance.get(key, 0) + 1
        summary = (
            f"{dataset.source_format.upper()} dataset with {len(dataset.session_ids)} sessions, "
            f"{len(dataset.feature_names)} stable features, and {sum(missing_values.values())} missing values."
        )
        return DatasetStatistics(
            feature_count=len(dataset.feature_names),
            row_count=len(dataset.session_ids),
            class_balance=class_balance,
            missing_values=missing_values,
            summary=summary,
        )

    def _session_signals(self, session: InterviewSession) -> Sequence[ExtractedSignal]:
        """Reuse stored signals when present, otherwise invoke SignalExtractor."""
        return session.extracted_signals or self._signal_extractor.extract(session)

    def _signal_row(self, signals: Sequence[ExtractedSignal]) -> List[float]:
        """Align signal values to the immutable feature ordering contract."""
        values = {signal.signal_name: signal.value for signal in signals}
        return [float(values.get(name, np.nan)) for name in self.feature_names]

    def _csv_signal_row(self, row: Dict[str, str]) -> List[float]:
        """Map only existing flattened generator columns to signal features."""
        columns = {
            "interview_duration_seconds": "duration_seconds",
            "transcript_turn_count": "num_turns",
            "total_llm_calls": "num_llm_calls",
            "total_tool_calls": "num_tool_calls",
            "average_llm_latency_ms": "avg_llm_latency_ms",
            "total_tokens": "total_tokens",
        }
        return [
            self._numeric_or_nan(row.get(columns[name], "")) if name in columns else np.nan
            for name in self.feature_names
        ]

    def _labels_from_sessions(self, sessions: Sequence[InterviewSession]) -> LabelSet:
        """Extract aligned ground-truth labels without reading predicted diagnosis."""
        return LabelSet(
            root_causes=tuple(
                self._enum_value(session.ground_truth.expected_failure_type)
                if session.ground_truth else None
                for session in sessions
            ),
            affected_components=tuple(
                self._enum_value(session.ground_truth.expected_affected_component)
                if session.ground_truth else None
                for session in sessions
            ),
            failure_categories=tuple(
                session.ground_truth.scenario_id if session.ground_truth else None
                for session in sessions
            ),
        )

    def _dataset(
        self,
        rows: Sequence[Sequence[float]],
        labels: LabelSet,
        session_ids: Sequence[str],
        source_format: str,
        issues: Sequence[LoadIssue],
    ) -> MLDataset:
        """Create a consistently shaped dataset, including for empty sources."""
        features = np.asarray(rows, dtype=float)
        if not rows:
            features = np.empty((0, len(self.feature_names)), dtype=float)
        return MLDataset(
            features=features,
            feature_names=self.feature_names,
            labels=labels,
            session_ids=tuple(session_ids),
            source_format=source_format,
            issues=tuple(issues),
        )

    def _record_issue(
        self, issues: List[LoadIssue], row_number: int, message: str, strict: bool
    ) -> None:
        """Append a deterministic issue or raise immediately in strict mode."""
        issue = LoadIssue(row_number=row_number, message=message)
        if strict:
            raise DatasetValidationError(f"Row {row_number}: {message}")
        issues.append(issue)

    def _numeric_or_nan(self, value: str) -> float:
        """Parse a CSV numeric value, retaining unavailable values as NaN."""
        try:
            return float(value) if value.strip() else np.nan
        except ValueError:
            return np.nan

    def _optional_value(self, value: Optional[str]) -> Optional[str]:
        """Normalize empty CSV label fields to an unavailable label."""
        return value or None

    def _enum_value(self, value: object) -> Optional[str]:
        """Serialize an optional schema enum without coupling to enum classes."""
        if value is None:
            return None
        return str(getattr(value, "value", value))
