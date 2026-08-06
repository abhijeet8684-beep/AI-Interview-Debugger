from __future__ import annotations

import pandas as pd
from app.ml_analytics import style_feature_drift_dataframe, style_label_drift_dataframe


def test_style_feature_drift_dataframe_handles_empty():
    df = pd.DataFrame([])
    styler = style_feature_drift_dataframe(df)
    # Should return a Styler without raising
    assert hasattr(styler, "render") or hasattr(styler, "to_html")


def test_style_feature_drift_dataframe_with_values_returns_styler():
    df = pd.DataFrame(
        [
            {"feature_name": "f1", "jensen_shannon_divergence": 0.05, "population_stability_index": 0.01, "shifted": False},
            {"feature_name": "f2", "jensen_shannon_divergence": 0.35, "population_stability_index": 0.20, "shifted": True},
        ]
    )
    styler = style_feature_drift_dataframe(df)
    assert hasattr(styler, "render") or hasattr(styler, "to_html")


def test_style_label_drift_dataframe_with_values_returns_styler():
    df = pd.DataFrame(
        [
            {"label_group": "A", "label": "a", "baseline_percentage": 10.0, "current_percentage": 12.0, "percentage_change": 2.0},
            {"label_group": "B", "label": "b", "baseline_percentage": 50.0, "current_percentage": 45.0, "percentage_change": -5.0},
        ]
    )
    styler = style_label_drift_dataframe(df)
    assert hasattr(styler, "render") or hasattr(styler, "to_html")


def test_styling_fallback_does_not_raise():
    # Create a DataFrame that will cause mapping functions to raise for some values
    df = pd.DataFrame(
        [{"feature_name": "f1", "jensen_shannon_divergence": "not_a_number", "population_stability_index": 0.1, "shifted": "maybe"}]
    )
    # Should not raise
    styler = style_feature_drift_dataframe(df)
    assert hasattr(styler, "render") or hasattr(styler, "to_html")
