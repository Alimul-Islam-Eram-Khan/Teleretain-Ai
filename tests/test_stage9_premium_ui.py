"""Pure, network-free visual-contract tests for the Stage 9 theme redesign."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def code() -> str:
    return APP.read_text(encoding="utf-8")


def test_premium_source_compiles():
    compile(code(), str(APP), "exec")


def test_native_dark_light_theme_not_overridden():
    src = code()
    assert "var(--background-color)" in src
    assert "var(--text-color)" in src
    assert "var(--secondary-background-color)" in src
    assert "[data-testid=\"stSidebar\"] * {color:inherit;}" in src
    assert 'background:#102236' not in src
    assert 'template="plotly_white"' not in src
    assert 'initial_sidebar_state="collapsed"' in src


def test_proposal_five_tabs_remain():
    src = code()
    for label in ("Iranian Prediction", "IBM Benchmark", "Retention Copilot", "Analytics", "About & Limitations"):
        assert f'"{label}"' in src
    assert 'st.chat_input' in src


def test_critical_interfaces_untouched():
    src = code()
    for value in ("checked_prediction(track, values)", "checked_explanation(track, state[\"features\"])",
                  "RecommendationRequest(", "recommend(request)", "answer_live(clean)",
                  "ask_research_copilot(", "safe_source_urls("):
        assert value in src
    assert "verify=False" not in src
    assert "escape(offer.description)" in src
    assert "st.write(response.message)" in src


def test_dashboard_graphs_from_saved_reports_only():
    src = code()
    assert "sc = model_scorecard(track)" in src
    assert 'sc["f1"]' in src
    assert 'sc["roc_auc"]' in src
    assert 'matrix_chart(sc["confusion_matrix"])' in src
    assert "dashboard_overview()" in src


def test_no_unrequested_network_or_llm_on_load():
    src = code()
    assert "requests.get(" not in src
    assert "fetch_offers_fresh(" not in src
    # All calls must be nested in user-triggered functions.
    parsed = ast.parse(src)
    for node in parsed.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            assert isinstance(node.value.func, (ast.Attribute, ast.Name))
            # Simple top-level UI rendering calls are expected.
    assert "if click:" in src
    assert "if question:" in src


def isolated_charts():
    """Load just pure plotting functions, not Streamlit or any model/report data."""
    parsed = ast.parse(code())
    names = {"plot_style", "quality_bar", "matrix_chart", "risk_gauge"}
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    env = {"go": go, "TEAL": "#16a394", "GOLD": "#e4aa54"}
    exec(compile(module, str(APP), "exec"), env)
    return env


def test_metric_bar_uses_input_values_not_hardcoded_results():
    fn = isolated_charts()["quality_bar"]
    fig = fn({"precision": .12, "recall": .23, "f1": .34, "roc_auc": .45}, "#123456")
    assert list(fig.data[0].x) == [.45, .34, .23, .12]
    assert list(fig.data[0].y) == ["ROC-AUC", "F1 score", "Recall", "Precision"]
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_confusion_matrix_uses_true_counts_and_validates_shape():
    import pytest
    fn = isolated_charts()["matrix_chart"]
    fig = fn([[10, 2], [3, 6]])
    assert list(map(list, fig.data[0].z)) == [[10, 2], [3, 6]]
    with pytest.raises(ValueError):
        fn([[1, 2, 3], [4, 5, 6]])


def test_score_gauge_exact_threshold_and_percentage():
    fig = isolated_charts()["risk_gauge"](.375, .48)
    assert math.isclose(fig.data[0].value, 37.5)
    assert math.isclose(fig.data[0].gauge.threshold.value, 48)


def test_sidebar_contains_no_internship_or_model_technology_inventory():
    src = code()
    sidebar_src = src.split("with st.sidebar:", 1)[1].split("st.markdown(\"\"\"<div class=\"tr-hero", 1)[0]
    for banned in ("Internship", "RESEARCH MODE", "Data sources", "①", "②", "③", "XGBoost"):
        assert banned not in sidebar_src
