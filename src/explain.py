"""TeleRetain AI Stage 4: SHAP explanations for locked Stage 3 models.

Public-data proof of concept: no Teletalk customer data, no causal claims.

Iranian RandomForestClassifier: sklearn TreeExplainer, positive-class
probability contributions (probability-point units).
IBM XGBClassifier: XGBoost's exact built-in TreeSHAP (pred_contribs=True),
raw-margin / log-odds contributions. Native method avoids older
SHAP/XGBoost base_score compatibility bugs. Both are TreeSHAP.

Never pass IBM features to Iranian model or vice versa; saved TrackInputAdapter
in the original Stage 3 Pipeline enforces the complete feature contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import DMatrix, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
TRACKS = frozenset(("iranian", "ibm"))


@dataclass
class LoadedTrack:
    name: str
    pipeline: Pipeline
    report: dict
    feature_names: list[str]
    space: str


def load_track(track: str) -> LoadedTrack:
    if track not in TRACKS:
        raise ValueError("Track must be 'iranian' or 'ibm'.")
    report_path = ROOT / "reports" / "stage3" / f"{track}_results.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Run Stage 3 first; missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("track") != track:
        raise ValueError(f"Stage 3 report track mismatch: {track}")
    model_path = ROOT / "models" / f"{track}_best_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    # Only load our own locally generated, verified artifacts. Untrusted pickle is unsafe.
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != [
        "input_adapter", "preprocessor", "model"
    ]:
        raise ValueError("Unexpected saved model structure; do not proceed.")
    adapter = pipeline.named_steps["input_adapter"]
    if adapter.track != track or adapter.feature_columns != report["input_features"]:
        raise ValueError("Model feature contract/track does not match Stage 3 report.")
    model = pipeline.named_steps["model"]
    expected_type = RandomForestClassifier if track == "iranian" else XGBClassifier
    expected_name = "random_forest" if track == "iranian" else "xgboost"
    if report["best_model_name"] != expected_name or not isinstance(model, expected_type):
        raise ValueError(
            f"This validated Stage 4 pack expects {track}={expected_name}. "
            "If you retrained/select a different model, update SHAP strategy after review."
        )
    names = [str(x) for x in pipeline.named_steps["preprocessor"].get_feature_names_out()]
    if len(names) != int(report["transformed_feature_count"]):
        raise ValueError("Preprocessing output dimension differs from Stage 3 report.")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate transformed feature names cannot be explained safely.")
    return LoadedTrack(track, pipeline, report, names,
                       "probability" if track == "iranian" else "log_odds")


def _validate_and_transform(ctx: LoadedTrack, raw_features: pd.DataFrame):
    # This first checks EXACT expected column names and correct track.
    checked = ctx.pipeline.named_steps["input_adapter"].transform(raw_features)
    transformed = ctx.pipeline.named_steps["preprocessor"].transform(checked)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float64)
    if transformed.ndim != 2 or transformed.shape != (len(checked), len(ctx.feature_names)):
        raise ValueError("Unexpected preprocessing output shape.")
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite transformed features; stop, review data quality.")
    return checked, transformed


def _tree_shap(ctx: LoadedTrack, transformed: np.ndarray):
    model = ctx.pipeline.named_steps["model"]
    if ctx.name == "iranian":
        explainer = shap.TreeExplainer(model, model_output="raw")
        raw_values = explainer.shap_values(transformed, check_additivity=True)
        if isinstance(raw_values, list):
            if len(raw_values) != 2:
                raise ValueError("Expected two Random Forest output classes.")
            values = np.asarray(raw_values[1], dtype=np.float64)
        else:
            all_values = np.asarray(raw_values, dtype=np.float64)
            if all_values.ndim != 3 or all_values.shape[-1] != 2:
                raise ValueError(f"Unexpected Random Forest SHAP shape: {all_values.shape}")
            values = all_values[:, :, 1]
        expected = np.asarray(explainer.expected_value, dtype=np.float64).ravel()
        if len(expected) != 2:
            raise ValueError(f"Unexpected Random Forest baseline shape: {expected.shape}")
        base = float(expected[1])
    else:
        # Native XGBoost TreeSHAP exactly matches this model's raw margin output;
        # unlike older shap.TreeExplainer releases it supports modern base_score.
        matrix = DMatrix(transformed.astype(np.float32, copy=False))
        contributions = np.asarray(
            model.get_booster().predict(matrix, pred_contribs=True), dtype=np.float64
        )
        if contributions.ndim != 2 or contributions.shape[1] != transformed.shape[1] + 1:
            raise ValueError(f"Unexpected XGBoost contributions shape: {contributions.shape}")
        values = contributions[:, :-1]
        baselines = contributions[:, -1]
        if not np.allclose(baselines, baselines[0], rtol=0, atol=1e-5):
            raise ValueError("Inconsistent XGBoost SHAP baselines.")
        base = float(baselines[0])
    if values.shape != transformed.shape or not np.isfinite(values).all() or not np.isfinite(base):
        raise ValueError("Invalid SHAP output shape or non-finite contribution.")
    reconstruction = base + values.sum(axis=1)
    reconstructed_prob = reconstruction if ctx.space == "probability" else expit(reconstruction)
    model_prob = np.asarray(model.predict_proba(transformed)[:, 1], dtype=np.float64)
    # Catch output-space mistakes instead of showing misleading attributions.
    if not np.allclose(reconstructed_prob, model_prob, atol=5e-5, rtol=1e-5):
        maximum = float(np.max(np.abs(reconstructed_prob - model_prob)))
        raise ValueError(f"SHAP additivity check failed; max probability gap: {maximum}")
    return values, base, model_prob


def explain_batch(track: str, raw_features: pd.DataFrame) -> dict[str, Any]:
    """Use *named original input features*; returns SHAP values and probabilities.

    Never call with target Churn, customerID, mixed-track features, or transformed
    arrays. These are forbidden by the Stage 3 TrackInputAdapter contract.
    """
    ctx = load_track(track)
    if not isinstance(raw_features, pd.DataFrame) or raw_features.empty:
        raise ValueError("Supply a non-empty DataFrame of original named features.")
    _, transformed = _validate_and_transform(ctx, raw_features)
    values, base, probs = _tree_shap(ctx, transformed)
    pipeline_prob = ctx.pipeline.predict_proba(raw_features)[:, 1]
    if not np.allclose(probs, pipeline_prob, atol=5e-5, rtol=1e-5):
        raise ValueError("SHAP features do not reproduce saved pipeline predictions.")
    return {
        "track": track,
        "data_source": ctx.report["data_source"],
        "model": ctx.report["best_model_name"],
        "feature_names": ctx.feature_names,
        "feature_data": transformed,
        "values": values,
        "base_value": base,
        "output_space": ctx.space,
        "probabilities": probs,
        "threshold": float(ctx.report["selected_threshold"]),
    }


def _local_from_batch(result: dict, row_index: int) -> dict:
    if not 0 <= row_index < len(result["probabilities"]):
        raise IndexError("SHAP row index out of bounds.")
    values = result["values"][row_index]
    order = np.argsort(np.abs(values))[::-1]
    return {
        "track": result["track"],
        "data_source": result["data_source"],
        "selected_model": result["model"],
        "probability_churn": float(result["probabilities"][row_index]),
        "decision_threshold": float(result["threshold"]),
        "predicted_class": int(result["probabilities"][row_index] >= result["threshold"]),
        "shap_output_space": result["output_space"],
        "shap_base_value": float(result["base_value"]),
        "additive_output": float(result["base_value"] + values.sum()),
        "feature_contributions": [
            {"feature": result["feature_names"][int(i)], "shap_value": float(values[int(i)])}
            for i in order
        ],
        "disclaimer": "SHAP describes this model's prediction; not a causal statement, not a Teletalk customer prediction.",
    }


def explain_one(track: str, raw_features: pd.DataFrame) -> dict:
    """One original-feature-row DataFrame -> plain JSON-compatible local explanation."""
    if not isinstance(raw_features, pd.DataFrame) or len(raw_features) != 1:
        raise ValueError("explain_one requires exactly ONE DataFrame row.")
    return _local_from_batch(explain_batch(track, raw_features), 0)


def plot_global(result: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    track = result["track"]
    mean_abs = np.mean(np.abs(result["values"]), axis=0)
    importance_order = np.argsort(mean_abs)[-18:]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(np.arange(len(importance_order)), mean_abs[importance_order])
    ax.set_yticks(np.arange(len(importance_order)))
    ax.set_yticklabels([result["feature_names"][i] for i in importance_order], fontsize=9)
    unit = "probability points" if result["output_space"] == "probability" else "log-odds units"
    ax.set_xlabel(f"Mean absolute TreeSHAP contribution ({unit})")
    ax.set_title(f"{track.title()} – Validation split, global TreeSHAP importance")
    fig.tight_layout()
    fig.savefig(output_dir / f"{track}_global_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # True SHAP summary/beeswarm plot, not an impurity-based importance chart.
    # The values are either probability contributions (RF) or log-odds (XGB).
    explanation = shap.Explanation(
        values=result["values"],
        base_values=np.full(len(result["values"]), result["base_value"]),
        data=result["feature_data"],
        feature_names=result["feature_names"],
    )
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=18, show=False, plot_size=(11, 8))
    fig = plt.gcf()
    fig.suptitle(f"{track.title()} – Validation TreeSHAP summary ({unit})", y=1.01)
    fig.savefig(output_dir / f"{track}_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_local(result: dict, row_index: int, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exp = shap.Explanation(
        values=result["values"][row_index],
        base_values=result["base_value"],
        data=result["feature_data"][row_index],
        feature_names=result["feature_names"],
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=13, show=False)
    fig = plt.gcf()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
