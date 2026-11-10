"""TeleRetain AI | design-system refresh for the existing verified Stage 9 app.

Pure UI overlay: all prediction, SHAP, catalog and LLM boundaries remain in
src.dashboard_stage9, src.retention_recommender and src.stage8_chat.
Run: python -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_stage9 import (
    TRACK_NAMES, checked_explanation, checked_prediction, input_form_fields,
    local_coverage_summary, model_scorecard, public_validation_count,
    public_validation_sample, safe_source_urls, sanitized_chat_question,
)

st.set_page_config(
    page_title="TeleRetain | Intelligence Workspace", page_icon="📊",
    layout="wide", initial_sidebar_state="collapsed",
    menu_items={"About": "TeleRetain | Public-source analytics workspace."},
)

# All light/dark surfaces & controls inherit Streamlit's actual active theme.
# Do not force a white body, dark text, or dark sidebar (the former Stage 9 bug).
STYLE = r"""
<style>
:root {
  --tr-teal:#0d9488; --tr-blue:#468ae8; --tr-gold:#e1a44b;
  --tr-border:color-mix(in srgb,var(--text-color) 13%,transparent);
  --tr-muted:color-mix(in srgb,var(--text-color) 73%,var(--background-color));
  --tr-card:color-mix(in srgb,var(--background-color) 94%,var(--text-color));
  --tr-soft:color-mix(in srgb,var(--background-color) 97%,var(--text-color));
}
[data-testid="stAppViewContainer"] {
  background:var(--background-color); color:var(--text-color);
}
[data-testid="stHeader"] {background:var(--background-color);}
[data-testid="stSidebar"] {
  background:var(--secondary-background-color); color:var(--text-color);
  border-right:1px solid var(--tr-border);
}
[data-testid="stSidebar"] * {color:inherit;}
.block-container {max-width:1440px; padding-top:1.35rem; padding-bottom:3rem;}
.tr-brand {display:flex;align-items:center;gap:13px;margin-bottom:1rem;}
.tr-mark {width:38px;height:38px;border-radius:12px;background:#0f766e;color:#fff;
 display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:850;
 box-shadow:0 5px 17px #0f766e35;}
.tr-brandname {font-size:1.15rem;font-weight:800;letter-spacing:-.035em;line-height:1.1;color:var(--text-color);}
.tr-brandline {font-size:.72rem;margin-top:3px;color:var(--tr-muted);}
.tr-hero {position:relative;overflow:hidden;isolation:isolate;border-radius:22px;
 background:linear-gradient(115deg,#0b172c 0%,#123954 60%,#0e736c 100%);
 padding:36px 42px 37px;color:#fff;margin:8px 0 19px;box-shadow:0 14px 40px #08172b24;}
.tr-hero:after {content:"";position:absolute;width:480px;height:480px;border:1px solid #ffffff20;
 border-radius:50%;right:-120px;top:-265px;box-shadow:0 0 0 70px #ffffff09,0 0 0 145px #ffffff07;z-index:-1;}
.tr-hero .kicker {text-transform:uppercase;font-weight:750;letter-spacing:.18em;font-size:.68rem;
 color:#85ead6;margin-bottom:15px;}
.tr-hero h1 {font-size:clamp(2.05rem,3.5vw,3.25rem)!important;line-height:1.05;
 font-weight:800;letter-spacing:-.055em;color:#fff!important;margin:0 0 13px!important;}
.tr-hero p {font-size:1.02rem;line-height:1.5;max-width:610px;color:#d5e8ef;margin:0;}
.tr-hero .hero-pills {margin-top:23px;display:flex;flex-wrap:wrap;gap:8px;}
.tr-hero .pill {color:#e6faf6;border:1px solid #ffffff30;border-radius:100px;
 padding:6px 12px;font-size:.75rem;background:#ffffff0e;}
.tr-section {display:flex;justify-content:space-between;align-items:baseline;gap:14px;margin:14px 0 10px;}
.tr-section h2 {font-size:1.22rem!important;font-weight:780!important;letter-spacing:-.025em;
 color:var(--text-color)!important;margin:0!important;}
.tr-section .aside {font-size:.78rem;color:var(--tr-muted);text-align:right;}
.tr-subtitle {font-size:.88rem;color:var(--tr-muted);margin:-3px 0 14px;line-height:1.48;}
.tr-note {border:1px solid var(--tr-border);background:var(--tr-soft);border-radius:10px;
 padding:10px 14px;font-size:.79rem;color:var(--tr-muted);margin:12px 0 16px;line-height:1.5;}
.tr-track {border:1px solid var(--tr-border);border-radius:16px;background:var(--tr-card);
 padding:16px 18px 6px;margin:0 0 4px;}
.tr-track .eyebrow {font-size:.7rem;text-transform:uppercase;letter-spacing:.13em;
 font-weight:750;color:var(--tr-muted);}
.tr-track .track-name {font-weight:780;letter-spacing:-.025em;font-size:1.18rem;
 margin-top:3px;color:var(--text-color);}
.tr-track .track-desc {color:var(--tr-muted);font-size:.79rem;margin-top:4px;}
.tr-statrow {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0 8px;}
.tr-stat {border-top:1px solid var(--tr-border);padding-top:10px;}
.tr-stat span {font-size:.74rem;color:var(--tr-muted);}
.tr-stat strong {display:block;font-size:1.55rem;letter-spacing:-.05em;line-height:1.2;
 color:var(--text-color);font-variant-numeric:tabular-nums;}
.tr-offer {border:1px solid var(--tr-border);border-radius:14px;padding:17px 19px;
 background:var(--tr-card);margin:0 0 10px;}
.tr-offer .offer-type {font-weight:760;color:color-mix(in srgb,var(--text-color) 48%,#16a394);letter-spacing:.13em;
 font-size:.67rem;text-transform:uppercase;margin-bottom:8px;}
.tr-offer .offer-name {font-weight:740;color:var(--text-color);font-size:1.04rem;line-height:1.45;}
.tr-offer .offer-info {color:var(--tr-muted);font-size:.88rem;margin-top:8px;}
.tr-offer .offer-price {color:var(--text-color);font-weight:780;font-size:1rem;}
.tr-foot {font-size:.76rem;color:var(--tr-muted);padding:12px 2px 22px;}
/* Restrict cosmetics to native surfaces; never override native text or icon contrast. */
[data-testid="stMetric"] {background:var(--tr-card);border:1px solid var(--tr-border);
 border-radius:14px;padding:13px 15px;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:10px;overflow-x:auto;}
[data-testid="stTabs"] button[role="tab"] {border-radius:9px 9px 0 0;
 font-weight:640;padding:12px 14px;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color:var(--text-color);}
div[data-testid="stButton"] button,div[data-testid="stFormSubmitButton"] button {border-radius:10px;}
@media(max-width:760px) {
 .block-container{padding-top:.8rem;}
 .tr-hero{border-radius:16px;padding:26px 23px;}
 .tr-hero h1{font-size:2.2rem!important;}
 .tr-section .aside{display:none;}
 .tr-brand{margin-bottom:.3rem;}
}
@media(prefers-reduced-motion:reduce){.tr-hero *{animation:none!important;transition:none!important;}}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

TEAL = "#16a394"
BLUE = "#4a90e2"
GOLD = "#e4aa54"


def heading(title: str, subtitle: str = "", aside: str = "") -> None:
    st.markdown(
        f'<div class="tr-section"><h2>{escape(title)}</h2>'
        f'<span class="aside">{escape(aside)}</span></div>', unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<div class="tr-subtitle">{escape(subtitle)}</div>', unsafe_allow_html=True)


def hint(text: str) -> None:
    st.markdown(f'<div class="tr-note">{escape(text)}</div>', unsafe_allow_html=True)


def controlled_error(exc: Exception) -> None:
    # Do not display arbitrary exception contents (API responses, file paths, keys).
    st.error(f"Action could not be completed ({type(exc).__name__}). Check local verification logs.")


def plot_style(fig: go.Figure, height: int = 190) -> go.Figure:
    """Transparent Plotly backgrounds follow Streamlit's light/dark chart theme."""
    fig.update_layout(
        height=height, margin=dict(l=4, r=26, t=8, b=14),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(size=12),
    )
    return fig


def quality_bar(sc: dict, color: str, height: int = 175) -> go.Figure:
    """Chart data is solely from the signed-off Stage 3 saved test metrics."""
    names = ["Precision", "Recall", "F1 score", "ROC-AUC"]
    values = [sc["precision"], sc["recall"], sc["f1"], sc["roc_auc"]]
    fig = go.Figure(go.Bar(
        y=names[::-1], x=values[::-1], orientation="h", marker_color=color,
        text=[f"{v:.3f}" for v in values[::-1]], textposition="outside",
        cliponaxis=False, hovertemplate="%{y} · %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.16], visible=False, fixedrange=True)
    fig.update_yaxes(showgrid=False, ticks="", fixedrange=True)
    return plot_style(fig, height)


def matrix_chart(values: list[list[int]]) -> go.Figure:
    if len(values) != 2 or any(len(r) != 2 for r in values):
        raise ValueError("Expected a 2×2 confusion matrix")
    fig = go.Figure(go.Heatmap(
        z=values, x=["Predicted stay", "Predicted churn"],
        y=["Actual stay", "Actual churn"],
        text=values, texttemplate="%{text}",
        textfont=dict(size=17, color="#112b38"),
        colorscale=[[0, "#e0f4f1"], [1, "#70c5b7"]], showscale=False,
        hovertemplate="%{y}<br>%{x}: %{z} records<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="bottom", showgrid=False)
    return plot_style(fig, 276)


def overview_card(track: str, color: str) -> None:
    sc = model_scorecard(track)
    name = "Iranian churn" if track == "iranian" else "IBM benchmark"
    subtitle = "UCI dataset" if track == "iranian" else "Fictional Telco dataset"
    st.markdown(
        f'<div class="tr-track"><div class="eyebrow">MODEL / {escape(track.upper())}</div>'
        f'<div class="track-name">{escape(name)}</div>'
        f'<div class="track-desc">{escape(subtitle)} · {escape(str(sc["model"]))}</div>'
        f'<div class="tr-statrow">'
        f'<div class="tr-stat"><span>F1 score</span><strong>{sc["f1"]:.3f}</strong></div>'
        f'<div class="tr-stat"><span>ROC-AUC</span><strong>{sc["roc_auc"]:.3f}</strong></div>'
        f'</div></div>', unsafe_allow_html=True,
    )
    st.plotly_chart(quality_bar(sc, color), use_container_width=True, theme="streamlit", config={"displayModeBar": False})


def dashboard_overview() -> None:
    heading("Model performance", "Frozen test-set diagnostics from the two independent benchmark datasets.", "MODEL QUALITY / FINAL TEST")
    left, right = st.columns(2, gap="large")
    for container, track, color in ((left, "iranian", TEAL), (right, "ibm", BLUE)):
        with container:
            try:
                overview_card(track, color)
            except (FileNotFoundError, KeyError, ValueError, TypeError, OSError):
                st.info(f"{track.upper()} model scorecard not available. Run Stage 9 verification.")
    st.caption("Performance figures are from separate public datasets; do not interpret them as Teletalk customer predictions.")


def render_form_input(track: str, field: dict, row_index: int):
    name, kind, value = field["name"], field["kind"], field["value"]
    key = f"feature_{track}_{row_index}_{name}"
    if kind == "category":
        return st.selectbox(name, field["options"], index=field["options"].index(value), key=key)
    if kind == "int":
        return int(st.number_input(name, value=int(value), min_value=0, step=1, key=key))
    if kind == "nullable_float":
        if name == "TotalCharges":
            missing = st.checkbox("TotalCharges not available", value=value is None, key=key + "_missing")
            if missing:
                return None
        return float(st.number_input(name, value=float(value) if value is not None else 0.0,
                                     min_value=0.0, step=0.1, key=key))
    return float(st.number_input(name, value=float(value), min_value=0.0, step=0.1, key=key))


def shap_chart(result: dict) -> go.Figure:
    df = pd.DataFrame(result["feature_contributions"][:12]).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df["shap_value"], y=df["feature"], orientation="h",
        marker_color=[TEAL if x >= 0 else GOLD for x in df["shap_value"]],
        hovertemplate="%{y} · %{x:+.4f}<extra></extra>",
    ))
    unit = "probability" if result["shap_output_space"] == "probability" else "log-odds"
    fig.update_xaxes(title_text=f"SHAP contribution ({unit})", zeroline=True, zerolinecolor="#8798a0")
    return plot_style(fig, 400)


def risk_gauge(value: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value * 100, 2),
        number={"valueformat": ".1f", "suffix": "%", "font": {"size": 36}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": TEAL},
               "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [0, threshold * 100], "color": "rgba(22,163,148,.10)"},
                         {"range": [threshold * 100, 100], "color": "rgba(228,170,84,.16)"}],
               "threshold": {"line": {"color": GOLD, "width": 4},
                             "thickness": .75, "value": threshold * 100}},
    ))
    return plot_style(fig, 230)


def prediction_tab(track: str) -> None:
    name = "Iranian churn model" if track == "iranian" else "IBM benchmark model"
    heading(name, "Load a public validation record, adjust the inputs and run the saved model.", "EXPLAINABLE PREDICTION")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        try:
            count = public_validation_count(track)
            selected = int(st.number_input("Validation record", min_value=0, max_value=max(0, count - 1),
                                           step=1, value=0, key=f"sample_{track}"))
            sample = public_validation_sample(track, selected)
            descriptors = input_form_fields(track, sample)
        except Exception as exc:
            controlled_error(exc)
            return
        with st.form(f"form_{track}_{selected}"):
            with st.expander("Input features", expanded=True):
                values = {}
                fields_l, fields_r = st.columns(2, gap="medium")
                for i, field in enumerate(descriptors):
                    with (fields_l if i % 2 == 0 else fields_r):
                        values[field["name"]] = render_form_input(track, field, selected)
            submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)
        if submitted:
            try:
                pred = checked_prediction(track, values)
                st.session_state[f"pred_{track}"] = {"prediction": pred, "features": values, "sample": selected}
                st.session_state.pop(f"shap_{track}", None)
            except Exception as exc:
                st.session_state.pop(f"pred_{track}", None)
                controlled_error(exc)
    with right:
        state = st.session_state.get(f"pred_{track}")
        if state is None:
            st.info("Select a record and run a prediction to display its score and explanation.")
            image = ROOT / "reports" / "stage4" / f"{track}_global_importance.png"
            if image.is_file():
                st.caption("Global drivers · public validation dataset")
                st.image(str(image), use_container_width=True)
        else:
            pred = state["prediction"]
            if pred.get("track") != track:
                st.error("Dataset mismatch; prediction blocked.")
                return
            st.caption(f"Public validation record #{state['sample']} · {pred['selected_model']}")
            st.plotly_chart(risk_gauge(pred["churn_probability"], pred["decision_threshold"]),
                            use_container_width=True, theme="streamlit", config={"displayModeBar": False})
            a, b = st.columns(2)
            a.metric("Decision", str(pred["predicted_label"]))
            b.metric("Threshold", f"{pred['decision_threshold']:.2f}")
            st.caption("Model-estimated score for an independent public benchmark, not a personal customer assessment.")
            if st.button("Explain with SHAP", key=f"shap_button_{track}", use_container_width=True):
                try:
                    st.session_state[f"shap_{track}"] = checked_explanation(track, state["features"])
                except Exception as exc:
                    controlled_error(exc)
            explanation = st.session_state.get(f"shap_{track}")
            if explanation is not None:
                if abs(float(explanation["probability_churn"]) - pred["churn_probability"]) > 5e-5:
                    st.error("SHAP and prediction do not match; explanation withheld.")
                else:
                    heading("Local feature contributions", aside="TREE SHAP")
                    st.plotly_chart(shap_chart(explanation), use_container_width=True, theme="streamlit",
                                    config={"displayModeBar": False})
                    unit = "probability points" if explanation["shap_output_space"] == "probability" else "log-odds"
                    st.caption(f"Units: {unit}. Feature contributions are not causal effects.")
                    with st.expander("Additivity diagnostic"):
                        st.json({"output_space": explanation["shap_output_space"],
                                 "base_value": explanation["shap_base_value"],
                                 "additive_output": explanation["additive_output"],
                                 "probability": explanation["probability_churn"]})


def official_links(urls) -> None:
    for index, url in enumerate(safe_source_urls(urls)):
        st.link_button("Open official source" if index == 0 else f"Official source {index + 1}", url)


def show_recommendations(result) -> None:
    status = result.status
    if status == "recommended":
        st.success(f"{len(result.items)} matching public offer(s) from the official page")
    elif status == "no_match":
        st.info("No matching listing in the checked official source. Other offers may exist.")
    elif status == "eligibility_verification_required":
        st.warning("SIM eligibility must be confirmed by Teletalk.")
    else:
        st.warning("Current offer information could not be verified; no price is recommended.")
    if status != "recommended":
        st.write(result.message)
    if result.fetched_at_utc:
        st.caption(f"Retrieved (UTC): {result.fetched_at_utc}")
    if result.items:
        for offer in result.items:
            validity = f"{offer.validity_days} days" if offer.validity_days is not None else "Validity not stated"
            st.markdown(
                f'<div class="tr-offer"><div class="offer-type">{escape(offer.category.upper())} · PUBLIC OFFER</div>'
                f'<div class="offer-name">{escape(offer.description)}</div>'
                f'<div class="offer-info"><span class="offer-price">৳{offer.price_bdt}</span>'
                f' &nbsp;·&nbsp; {escape(validity)} &nbsp;·&nbsp; {escape(offer.ussd)}</div></div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"Source evidence · Offer #{offer.rank}"):
                st.write(offer.reason)
                st.code(offer.exact_source_row)
                st.caption("Personal activation and SIM eligibility not verified.")
    official_links(result.citations)
    st.caption(result.disclaimer)


def offer_finder() -> None:
    from src.retention_recommender import RecommendationRequest, recommend
    heading("Find the right public offer", "Compare live website listings by cost, validity and usage.", "LIVE OFFERS")
    with st.form("offer_query"):
        a, b, c = st.columns(3, gap="medium")
        with a:
            category = st.selectbox("Category", ["internet", "combo", "minute", "sms", "unlimited_weekend"])
            sim_type = st.selectbox("SIM type", ["public", "gen_z", "agami", "bornomala", "oporajita",
                                                 "mayer_hasi", "professional", "closed_sim"])
            budget = st.number_input("Budget (৳)", min_value=0, max_value=100000, value=100, step=10,
                                     help="Set to zero for no limit.")
        with b:
            exact_days = st.number_input("Validity · exact days", min_value=0, max_value=366, value=7, step=1,
                                         help="Zero means any validity.")
            min_days = st.number_input("Validity · minimum days", min_value=0, max_value=366, value=0, step=1)
            sort_by = st.selectbox("Rank by", ["lowest_price", "most_data", "longest_validity", "data_per_taka"])
        with c:
            data_mb = st.number_input("Minimum data · MB", min_value=0, max_value=500000, value=0, step=100)
            minutes = st.number_input("Minimum minutes", min_value=0, max_value=50000, value=0, step=10)
            sms = st.number_input("Minimum SMS", min_value=0, max_value=50000, value=0, step=10)
        click = st.form_submit_button("Search verified offers", type="primary", use_container_width=True)
    if click:
        try:
            if sort_by == "data_per_taka" and category not in ("internet", "combo"):
                raise ValueError("Data-per-taka applies only to Internet/Combo")
            request = RecommendationRequest(
                category=category, sim_type=sim_type, budget_bdt=budget or None,
                exact_validity_days=exact_days or None, min_validity_days=min_days or None,
                min_data_mb=data_mb or None, min_minutes=minutes or None,
                min_sms=sms or None, sort_by=sort_by, limit=5, language="banglish",
            )
            with st.spinner("Checking the official website…"):
                st.session_state["stage9_recs"] = recommend(request)
        except Exception as exc:
            controlled_error(exc)
    if st.session_state.get("stage9_recs") is not None:
        show_recommendations(st.session_state["stage9_recs"])


def official_services() -> None:
    heading("Services & SIM conditions", "Read public, source-linked guidance directly from the operator website.")
    with st.form("service_search"):
        q = st.text_input("Search official information", placeholder="e.g. Gen-Z eligibility conditions")
        go = st.form_submit_button("Search official website")
    if go:
        try:
            from src.teletalk_live import answer_live
            clean = sanitized_chat_question(q)
            with st.spinner("Checking official pages…"):
                st.session_state["service_answer"] = answer_live(clean)
        except Exception as exc:
            controlled_error(exc)
    response = st.session_state.get("service_answer")
    if response is not None:
        if response.mode.startswith("live"):
            st.success("Official source retrieved")
        else:
            st.warning("Current source information could not be verified")
        st.write(response.message)
        official_links(response.citations)


def copilot_tab() -> None:
    heading("Retention copilot", "A focused workspace for benchmark explanations and source-backed public offers.")
    with st.container():
        chat, offers, services = st.tabs(["Conversation", "Offer finder", "Services & eligibility"])
        with chat:
            a, b = st.columns(2, gap="medium")
            with a:
                track = st.selectbox("Model context", ["iranian", "ibm"],
                                     format_func=lambda t: TRACK_NAMES[t], key="chat_track")
            with b:
                live_llm = st.toggle("AI explanation (Gemini → Groq)", value=False,
                                     help="Uses your privately configured providers. Off by default.")
            features = st.session_state.get(f"pred_{track}", {}).get("features")
            if features is None:
                st.caption("To discuss a prediction, run a public validation record in its model tab first.")
            history = st.session_state.setdefault("chat_history", [])
            for message in history:
                with st.chat_message(message["role"]):
                    st.markdown(message["body"])  # Markdown escapes embedded HTML by default.
                    if message.get("citations"):
                        official_links(message["citations"])
            question = st.chat_input("Ask about churn, bundles or SIM conditions…")
            if question:
                try:
                    clean = sanitized_chat_question(question)
                    from src.stage8_chat import ask_research_copilot
                    with st.spinner("Preparing a source-checked answer…"):
                        mode, response, urls = ask_research_copilot(
                            clean, track=track, features=features, live_llm=live_llm,
                        )
                    citations = safe_source_urls(urls)
                    history.extend([{"role": "user", "body": clean},
                                    {"role": "assistant", "body": response,
                                     "citations": citations, "mode": mode}])
                    st.rerun()
                except Exception as exc:
                    controlled_error(exc)
            if history and st.button("Clear conversation", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()
            hint("Use public benchmark data only. Never enter real phone numbers, NIDs or private subscriber records.")
        with offers:
            offer_finder()
        with services:
            official_services()


def analytics_tab() -> None:
    heading("Model evaluation", "Saved Stage 3 test results · no retraining or invented measurements.", "ANALYTICS")
    for track, color in (("iranian", TEAL), ("ibm", BLUE)):
        name = "Iranian churn" if track == "iranian" else "IBM benchmark"
        try:
            sc = model_scorecard(track)
        except Exception as exc:
            controlled_error(exc)
            continue
        st.markdown(f"### {name}")
        a, b, c, d = st.columns(4)
        a.metric("F1 score", f"{sc['f1']:.3f}")
        b.metric("ROC-AUC", f"{sc['roc_auc']:.3f}")
        c.metric("Precision", f"{sc['precision']:.3f}")
        d.metric("Recall", f"{sc['recall']:.3f}")
        col_bar, col_matrix = st.columns([1.12, .88], gap="large")
        with col_bar:
            st.caption("Classification metrics · held-out test split")
            st.plotly_chart(quality_bar(sc, color, 280), use_container_width=True,
                            theme="streamlit", config={"displayModeBar": False})
            st.caption(f"{sc['model']} · threshold {sc['threshold']:.2f} · {sc['test_rows']} test records")
        with col_matrix:
            st.caption("Confusion matrix · held-out test split")
            st.plotly_chart(matrix_chart(sc["confusion_matrix"]), use_container_width=True,
                            theme="streamlit", config={"displayModeBar": False})
        st.divider()
    with st.expander("Candidate validation reports"):
        for track in ("iranian", "ibm"):
            path = ROOT / "reports" / "stage3" / f"{track}_candidate_validation_metrics.csv"
            if path.is_file():
                st.markdown(f"**{TRACK_NAMES[track]}**")
                st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)
    status = ROOT / "reports" / "stage3" / "iranian_status_sensitivity.json"
    if status.is_file():
        with st.expander("Iranian Status-feature sensitivity · diagnostic"):
            st.json(json.loads(status.read_text(encoding="utf-8")))
    st.caption("These independent benchmark metrics do not establish performance on real Teletalk subscribers.")


def about_tab() -> None:
    heading("About TeleRetain", "How the workspace handles predictions, public offers and evidence.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Prediction**")
        st.write("Two separate public benchmark models. Predictions and SHAP explanations never share input contracts.")
        st.markdown("**Offer intelligence**")
        st.write("On-demand official Teletalk public pages and deterministic offer matching. Personal SIM eligibility cannot be verified here.")
    with b:
        st.markdown("**AI assistance**")
        st.write("Optional Gemini/Groq explanations. Model scores and listed offer prices always come from their respective validated sources.")
        st.markdown("**Data protection**")
        st.write("Do not submit confidential subscriber data. This workspace does not include operator authentication or production security controls.")
    st.divider()
    heading("Source coverage", "Review status is not the same as exhaustive, current official coverage.")
    coverage = local_coverage_summary()
    if coverage["available"]:
        a, b, c, d = st.columns(4)
        a.metric("Approved offers", coverage["approved_offers"])
        b.metric("Reviewed pages", coverage["reviewed_entries"])
        c.metric("Pending review", coverage["review_queue"])
        d.metric("Conflicts", coverage["conflicts"])
        if coverage.get("discovery") is not None:
            with st.expander("Website discovery status"):
                st.json(coverage["discovery"])
    else:
        st.info("Local coverage audit unavailable.")
    with st.expander("Methodology & limitations"):
        st.markdown("""
        - Churn models are trained and tested on **public Iranian and IBM datasets**, not Teletalk customers.
        - SHAP features explain fitted model output, not causal churn drivers.
        - Published offers do not prove account-level activation, SIM eligibility or retention uplift.
        - The manually reviewed catalog is not a guarantee of complete or current coverage.
        - Quarantined conflicting records remain excluded until individually reviewed.
        - Live API calls are optional and may have rate limits; user questions may be sent to enabled providers.
        - This prototype has no production authentication, monitoring or real-subscriber data safeguards.
        """)


with st.sidebar:
    st.markdown("""<div class="tr-brand"><span class="tr-mark">◈</span>
        <div><div class="tr-brandname">TeleRetain</div>
        <div class="tr-brandline">Intelligence workspace</div></div></div>""", unsafe_allow_html=True)
    st.caption("Public-data research environment")
    st.divider()
    st.caption("Appearance follows Settings → Theme")

st.markdown("""<div class="tr-hero"><div class="kicker">TELECOM DECISION INTELLIGENCE</div>
<h1>Signals into clarity.</h1>
<p>Understand churn patterns, explore model drivers, and discover source-backed public offers—within one focused workspace.</p>
<div class="hero-pills"><span class="pill">Explainable models</span>
<span class="pill">Official public offers</span><span class="pill">Source-linked insights</span></div></div>""",
unsafe_allow_html=True)

dashboard_overview()
st.divider()
# Keep the exact five proposal-approved main tabs; the language here is intentional.
pred_iran, pred_ibm, copilot, analytics, about = st.tabs([
    "Iranian Prediction", "IBM Benchmark", "Retention Copilot",
    "Analytics", "About & Limitations",
])
with pred_iran:
    prediction_tab("iranian")
with pred_ibm:
    prediction_tab("ibm")
with copilot:
    copilot_tab()
with analytics:
    analytics_tab()
with about:
    about_tab()

st.markdown("<div class='tr-foot'>TeleRetain · Public-source analytics workspace · Independent model datasets · Source-aware recommendations</div>",
            unsafe_allow_html=True)
