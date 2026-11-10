"""SHAP reports on VALIDATION data, avoiding repeated test-set exploration."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.explain import explain_batch, _local_from_batch, plot_global, plot_local, load_track

OUT = ROOT / "reports" / "stage4"


def run(track: str, max_rows: int):
    ctx = load_track(track)
    validation = pd.read_csv(ROOT / "data" / "processed" / track / "val.csv")
    feature_columns = ctx.report["input_features"]
    if len(validation) < 2 or max_rows < 2:
        raise ValueError("Validation sample size must be at least 2.")
    # Fixed random selection from validation ONLY. No training/refitting/test usage.
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(validation), min(max_rows, len(validation)), replace=False))
    selected = validation.iloc[indices]
    result = explain_batch(track, selected[feature_columns])
    OUT.mkdir(parents=True, exist_ok=True)
    plot_global(result, OUT)
    # One example predicted as churn and one predicted as non-churn when present.
    binary_pred = result["probabilities"] >= result["threshold"]
    picks = {}
    for label, value in (("churn", True), ("non_churn", False)):
        matches = np.flatnonzero(binary_pred == value)
        if len(matches):
            index = int(matches[0])
            explanation = _local_from_batch(result, index)
            explanation["validation_row_index"] = int(indices[index])
            file = OUT / f"{track}_local_{label}.json"
            file.write_text(json.dumps(explanation, indent=2, allow_nan=False), encoding="utf-8")
            plot_local(result, index, OUT / f"{track}_local_{label}_waterfall.png")
            picks[label] = file.name
    mean_importance = np.mean(np.abs(result["values"]), axis=0)
    ranking = np.argsort(mean_importance)[::-1]
    summary = {
        "track": track,
        "data_source": result["data_source"],
        "dataset_split_explained": "validation only (sampled with random seed 42)",
        "sample_size": int(len(indices)),
        "best_model": result["model"],
        "classification_threshold_locked_from_stage3": result["threshold"],
        "shap_method": "TreeExplainer (scikit-learn Random Forest)" if track == "iranian" else "XGBoost native exact TreeSHAP pred_contribs",
        "shap_output_space": result["output_space"],
        "shap_baseline": float(result["base_value"]),
        "global_mean_absolute_contributions": [
            {"feature": result["feature_names"][int(i)], "mean_absolute_shap": float(mean_importance[int(i)])}
            for i in ranking
        ],
        "local_examples": picks,
        "notes": [
            "The selected models and Stage 3 decision thresholds were not retrained or changed.",
            "SHAP attributions explain the model, not causality.",
            "Categorical features were one-hot encoded: each resulting column has a separate contribution.",
            "Explanations are on the processed feature scale, not the original data units.",
            "IBM XGBoost SHAP sums in log-odds and must be converted through sigmoid to get churn probability.",
            "Predictions are NOT for real Teletalk customers.",
            "No test-set labels or probabilities are used in this explanatory report.",
        ],
    }
    (OUT / f"{track}_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"PASS: {track} | {len(indices)} validation rows | "
          f"{result['values'].shape[1]} TreeSHAP columns | "
          f"output={result['output_space']} | global + local saved")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["iranian", "ibm", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=160,
                        help="Maximum validation rows per track for SHAP; default 160")
    args = parser.parse_args()
    tracks = ("iranian", "ibm") if args.track == "both" else (args.track,)
    for track in tracks:
        run(track, args.sample_size)
    print("Stage 4 finished. Next: python scripts/verify_stage4.py")


if __name__ == "__main__":
    main()
