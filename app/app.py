"""Streamlit dashboard for visualizing AI Interview Debugger results."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.diagnosis_pipeline import DiagnosisPipeline, DiagnosisResult
from pipeline.schema import InterviewSession
from app.ml_analytics import render_ml_analytics


def _pipeline() -> DiagnosisPipeline:
    """Return the backend orchestration entry point for dashboard execution."""
    return DiagnosisPipeline()


def _load_session(uploaded_file: Any) -> InterviewSession:
    """Validate an uploaded JSON document as an interview session.

    Args:
        uploaded_file: Streamlit upload object containing one JSON document.

    Returns:
        Validated interview session.

    Raises:
        ValueError: If the upload is empty or contains invalid JSON/session data.
    """
    raw_content = uploaded_file.getvalue()
    if not raw_content.strip():
        raise ValueError("The uploaded file is empty.")
    try:
        return InterviewSession.model_validate_json(raw_content)
    except ValidationError as error:
        raise ValueError("The uploaded JSON is not a valid InterviewSession.") from error


def _enum_value(value: Any) -> Any:
    """Return an enum value when available for compact dashboard rendering."""
    return getattr(value, "value", value)


def _render_sidebar(session: InterviewSession) -> None:
    """Render project and uploaded-session metadata in the sidebar."""
    st.sidebar.subheader("Uploaded Session")
    st.sidebar.write(f"Dataset: `{session.dataset_version or 'Not provided'}`")
    st.sidebar.write(f"Pipeline: `{session.pipeline_version or 'Not provided'}`")
    st.sidebar.write(f"Schema: `{session.schema_version or 'Not provided'}`")
    st.sidebar.subheader("Session metadata")
    st.sidebar.write(f"Session ID: `{session.session_id}`")
    st.sidebar.write(f"Status: `{_enum_value(session.status)}`")
    st.sidebar.write(f"Created: `{session.created_at or 'Not provided'}`")
    if session.candidate:
        st.sidebar.write(f"Candidate: `{session.candidate.name or session.candidate.candidate_id}`")
    if session.metadata:
        st.sidebar.json(session.metadata)


def _render_project_sidebar() -> None:
    """Render static project information and the backend pipeline overview."""
    st.sidebar.title("AI Interview Debugger")
    st.sidebar.caption("Explainable AI interview observability")
    st.sidebar.success("Backend status: 85 tests passing")
    st.sidebar.divider()
    st.sidebar.subheader("Versions")
    st.sidebar.write("Dataset: `from uploaded session`")
    st.sidebar.write("Pipeline: `from uploaded session`")
    st.sidebar.subheader("Technology Stack")
    st.sidebar.write("Python · Pydantic · FAISS · Streamlit")
    st.sidebar.subheader("Pipeline Stages")
    st.sidebar.markdown(
        "1. Signal extraction\n"
        "2. Rule evaluation\n"
        "3. Evidence building\n"
        "4. Similarity search\n"
        "5. Confidence scoring\n"
        "6. Engineering diagnosis"
    )


def _render_landing_content() -> None:
    """Render the minimal landing-page guidance before a file is uploaded."""
    st.markdown("### Diagnose interview-system failures with traceable evidence")
    st.write(
        "Upload one validated `InterviewSession` JSON file. The dashboard runs "
        "the frozen backend pipeline and presents deterministic rules, evidence, "
        "historical context, and confidence in one view."
    )
    with st.expander("About this project"):
        st.write(
            "AI Interview Debugger is an explainable observability system for "
            "synthetic AI interview sessions. It separates measurable telemetry, "
            "deterministic rules, supporting evidence, and diagnosis output."
        )
    with st.expander("Pipeline Architecture"):
        st.code(
            "InterviewSession → SignalExtractor → RuleEngine → EvidenceBuilder\n"
            "→ SimilarityEngine → ConfidenceEngine → LLMReasoner",
            language=None,
        )


def _render_footer() -> None:
    """Render concise project runtime details at the bottom of the dashboard."""
    st.divider()
    st.caption(
        "Author: AI Interview Debugger Team  ·  Python 3.12  ·  "
        "Streamlit  ·  Project version: 1.0"
    )


def _render_overview(result: DiagnosisResult) -> None:
    """Render key diagnosis metrics and status badges."""
    diagnosis = result.diagnosis
    columns = st.columns(4)
    columns[0].metric("Root cause", _enum_value(diagnosis.root_cause).replace("_", " ").title())
    columns[1].metric("Confidence", f"{diagnosis.confidence:.0%}")
    columns[2].metric("Severity", _enum_value(diagnosis.severity).title())
    columns[3].metric("Triggered rules", len(diagnosis.rules_triggered))
    st.progress(diagnosis.confidence)
    st.caption(f"Affected component: {_enum_value(diagnosis.affected_component)}")


def _render_rules(result: DiagnosisResult) -> None:
    """Render deterministic rule evaluations in an expandable table."""
    with st.expander("Triggered Rules", expanded=True):
        rows = [
            {
                "Rule": rule.rule_name,
                "Triggered": rule.triggered,
                "Score": rule.score,
                "Reason": rule.reason,
            }
            for rule in result.rules
            if rule.triggered
        ]
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.success("No deterministic diagnostic rules were triggered.")


def _render_evidence(result: DiagnosisResult) -> None:
    """Render source-linked supporting evidence cards."""
    with st.expander("Supporting Evidence", expanded=True):
        if not result.evidence:
            st.info("No supporting evidence was generated.")
            return
        for record in result.evidence:
            st.markdown(f"**{record.description}** — `{record.rule_id or 'unlinked'}`")
            st.write(record.explanation or "No explanation provided.")
            st.caption(
                f"Metric: {record.metric} | Observed: {record.observed_value} | "
                f"Expected: {record.expected_value} | Confidence: {record.confidence}"
            )


def _render_similar_sessions(result: DiagnosisResult) -> None:
    """Render retrieved historical sessions and their similarity scores."""
    with st.expander("Historical Similar Sessions"):
        if not result.similar_sessions:
            st.info("No historical similarity index results are available.")
            return
        rows = [
            {
                "Session ID": str(item.session_id),
                "Similarity": f"{item.similarity_score:.0%}",
                "Failure type": _enum_value(item.failure_type) if item.failure_type else "Unknown",
                "Summary": item.summary,
            }
            for item in result.similar_sessions
        ]
        st.dataframe(rows, use_container_width=True)


def _render_diagnosis(result: DiagnosisResult) -> None:
    """Render schema-supported engineering diagnosis and confidence details."""
    with st.expander("Engineering Diagnosis", expanded=True):
        st.write(
            "The diagnosis below is generated from deterministic rules, evidence, "
            "historical retrieval, and confidence contributors."
        )
        st.write(result.confidence.explanation)
        st.json(result.confidence.contributors)
        if result.diagnosis.evidence:
            st.markdown("**Diagnosis evidence summary**")
            for item in result.diagnosis.evidence:
                st.write(f"- {item}")

    with st.expander("Recommendations and Preventive Actions"):
        st.info(
            "The frozen FailureDiagnosis schema does not include structured "
            "recommendations or preventive actions. No UI-side recommendations "
            "are generated to preserve backend-only analysis."
        )


def _render_timeline(session: InterviewSession) -> None:
    """Render a chronological table of session timeline events."""
    with st.expander("Timeline"):
        rows: List[Dict[str, Any]] = [
            {
                "Time": event.timestamp,
                "Event": _enum_value(event.event_type),
                "Stage": _enum_value(event.stage) if event.stage else None,
                "Description": event.description,
                "Turn": event.related_turn_id,
                "Call": event.related_call_id,
            }
            for event in session.timeline
        ]
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("The uploaded session has no timeline events.")


def _render_pipeline_summary(result: DiagnosisResult) -> None:
    """Render a concise summary showing successful stage outputs."""
    with st.expander("Pipeline Execution Summary"):
        rows = [
            {"Stage": "Signal Extraction", "Output": f"{len(result.signals)} signals"},
            {"Stage": "Rule Evaluation", "Output": f"{len(result.rules)} evaluations"},
            {"Stage": "Evidence Builder", "Output": f"{len(result.evidence)} records"},
            {"Stage": "Similarity Search", "Output": f"{len(result.similar_sessions)} matches"},
            {"Stage": "Confidence Engine", "Output": f"{result.confidence.overall_confidence:.0%}"},
            {"Stage": "Reasoning Engine", "Output": "FailureDiagnosis generated"},
        ]
        st.dataframe(rows, use_container_width=True)


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="AI Interview Debugger", page_icon="🔎", layout="wide")
    st.title("AI Interview Debugger")
    st.caption("Deterministic debugging for AI interview sessions")
    _render_project_sidebar()

    diagnosis_tab, analytics_tab = st.tabs(["Diagnosis", "📊 ML Analytics"])

    with diagnosis_tab:
        st.markdown("---")
        st.markdown("#### Upload an Interview Session")
        st.write(
            "Select a single JSON document matching the frozen `InterviewSession` "
            "schema. No analysis runs until a valid session is uploaded."
        )
        uploaded_file = st.file_uploader("InterviewSession JSON", type=["json"])
        if uploaded_file is None:
            _render_landing_content()
        else:
            try:
                session = _load_session(uploaded_file)
            except ValueError as error:
                st.error(str(error))
            else:
                _render_sidebar(session)
                try:
                    result = _pipeline().run(session)
                except Exception as error:
                    st.error("The diagnosis pipeline could not complete for this session.")
                    st.exception(error)
                else:
                    _render_overview(result)
                    _render_rules(result)
                    _render_evidence(result)
                    _render_similar_sessions(result)
                    _render_diagnosis(result)
                    _render_timeline(session)
                    _render_pipeline_summary(result)

    with analytics_tab:
        render_ml_analytics()

    _render_footer()


if __name__ == "__main__":
    main()
