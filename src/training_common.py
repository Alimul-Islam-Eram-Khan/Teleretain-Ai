"""Stage 3: evaluate 3 algorithms independently on each public-data track.

Best algorithm ranked by validation average precision (PR-AUC/AP); threshold
selected on validation F1. Test remains untouched until model+threshold fixed.
No Teletalk subscriber records and no generated customer records are used.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import copy
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.data_prep_common import ROOT, make_preprocessor
from src.model_inputs import TrackInputAdapter

REPORTS = ROOT / "reports" / "stage3"
MODELS = ROOT / "models"
SEED = 42
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def get_candidates():
    # No class weighting: we will tune the operating threshold on validation.
    return OrderedDict([
        ("logistic_regression", LogisticRegression(max_iter=4000, solver="lbfgs", random_state=SEED)),
        ("random_forest", RandomForestClassifier(n_estimators=220, min_samples_leaf=2,
                                                    n_jobs=2, random_state=SEED)),
        ("xgboost", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  n_estimators=220, max_depth=3, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9,
                                  tree_method="hist", n_jobs=2, random_state=SEED)),
    ])


def _load_track(track: str):
    meta_file = ROOT / "reports" / "stage2" / f"{track}_data_quality.json"
    assert meta_file.is_file(), f"Missing Stage 2 report: {meta_file}"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["track"] == track, "Track metadata mismatch"

    raw_path = ROOT / meta["raw_file"]
    assert raw_path.is_file() and _sha256(raw_path) == meta["raw_sha256"], (
        f"Raw {track} CSV has changed since Stage 2; review and rerun Stage 2 first."
    )
    data = {}
    for split in ("train", "val", "test"):
        path = ROOT / "data" / "processed" / track / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Stage 2 split missing: {path}")
        frame = pd.read_csv(path)
        if len(frame) != meta["splits"][split]["rows"]:
            raise ValueError(f"{track}/{split} row count has changed; rerun Stage 2.")
        if set(frame["Churn"].dropna().unique()) != {0, 1}:
            raise ValueError(f"{track}/{split} has invalid/one-class churn labels")
        data[split] = frame
    pre_file = MODELS / f"{track}_preprocessor.joblib"
    if not pre_file.is_file():
        raise FileNotFoundError(f"Missing train-fitted preprocessor: {pre_file}")
    preprocessor = joblib.load(pre_file)  # use ONLY local project artifacts
    check_is_fitted(preprocessor)
    if list(preprocessor.feature_names_in_) != list(meta["feature_columns"]):
        raise ValueError(f"{track} preprocessor feature contract differs from Stage 2 metadata")
    if len(preprocessor.get_feature_names_out()) != meta["transformed_feature_count"]:
        raise ValueError("Transformed feature count differs from Stage 2 metadata")
    return meta, data, preprocessor


def _metrics(y_true, p, threshold):
    y_hat = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc_average_precision": float(average_precision_score(y_true, p)),
        "brier_score": float(brier_score_loss(y_true, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _best_threshold(y, probs):
    rows = [(float(t), float(f1_score(y, probs >= t, zero_division=0))) for t in THRESHOLDS]
    # Deterministic tie breaker: threshold closest to 0.5, then smallest.
    return sorted(rows, key=lambda r: (-r[1], abs(r[0] - 0.5), r[0]))[0][0], rows


def _save_candidate_chart(track: str, rows: list[dict]):
    names = [r["model"].replace("_", " ").title() for r in rows]
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(positions - 0.17, [r["pr_auc_average_precision"] for r in rows], 0.34, label="PR-AUC / AP")
    ax.bar(positions + 0.17, [r["roc_auc"] for r in rows], 0.34, label="ROC-AUC")
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.set_title(f"{track.title()} — validation comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / f"{track}_candidate_comparison.png", dpi=150)
    plt.close(fig)


def _save_threshold_chart(track: str, threshold_scores, choice: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([v[0] for v in threshold_scores], [v[1] for v in threshold_scores])
    ax.axvline(choice, linestyle="--", label=f"Selected threshold: {choice:.2f}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Validation F1")
    ax.set_title(f"{track.title()} — F1 threshold selection (validation only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / f"{track}_threshold_validation.png", dpi=150)
    plt.close(fig)


def _save_test_charts(track: str, y_test, probs, threshold: float):
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{track.title()} — untouched test split")
    plt.tight_layout()
    plt.savefig(REPORTS / f"{track}_test_confusion_matrix.png", dpi=150)
    plt.close()

    true_rate, avg_prob = calibration_curve(y_test, probs, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(avg_prob, true_rate, marker="o", label="Chosen model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn frequency")
    ax.set_title(f"{track.title()} — test reliability curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / f"{track}_test_calibration.png", dpi=150)
    plt.close(fig)


def _status_sensitivity(best_name, train, val, meta, full_val_metrics):
    """Iranian only: train a fresh no-Status model; validation comparison ONLY."""
    numeric = [x for x in meta["numeric_columns"] if x != "Status"]
    categorical = [x for x in meta["categorical_columns"] if x != "Status"]
    without_status = numeric + categorical
    pre = make_preprocessor(numeric, categorical)
    pre.fit(train[without_status])  # fit on train ONLY
    candidate = copy.deepcopy(get_candidates()[best_name])
    candidate.fit(pre.transform(train[without_status]), train["Churn"].to_numpy())
    probs = candidate.predict_proba(pre.transform(val[without_status]))[:, 1]
    threshold, _ = _best_threshold(val["Churn"].to_numpy(), probs)
    no_status_metrics = _metrics(val["Churn"].to_numpy(), probs, threshold)
    return {
        "purpose": "Diagnostic validation-only comparison; not proof of temporal leakage or causality",
        "algorithm": best_name,
        "full_model_validation": full_val_metrics,
        "without_status_validation": no_status_metrics,
        "difference_in_validation_average_precision_full_minus_no_status": round(
            full_val_metrics["pr_auc_average_precision"] - no_status_metrics["pr_auc_average_precision"], 6
        ),
        "notes": "Same training/validation rows. Neither model is fitted on test; test metrics for selected full model are recorded separately.",
    }


def train_track(track: str) -> dict:
    if track not in ("iranian", "ibm"):
        raise ValueError("Only iranian or ibm allowed")
    meta, data, preprocessor = _load_track(track)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    features = meta["feature_columns"]
    adapter = TrackInputAdapter(track, features)
    # Uses the EXACT training-fitted Stage 2 preprocessing, without refitting it.
    X_train = preprocessor.transform(adapter.transform(data["train"][features]))
    X_val = preprocessor.transform(adapter.transform(data["val"][features]))
    y_train = data["train"]["Churn"].to_numpy()
    y_val = data["val"]["Churn"].to_numpy()
    if X_train.shape[1] != meta["transformed_feature_count"]:
        raise ValueError("Unexpected transformed training feature count")

    # Guard against silently all-zero SeniorCitizen one-hot encoding on IBM.
    if track == "ibm":
        cat_pre = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        cat_cols = meta["categorical_columns"]
        index = cat_cols.index("SeniorCitizen")
        expected = {"0", "1"}
        observed = {str(k) for k in cat_pre.categories_[index]}
        if not expected.issubset(observed):
            raise ValueError(f"IBM SeniorCitizen encoding mismatch: observed={observed}")

    candidate_results = []
    validation_probs = {}
    threshold_curves = {}
    fitted_models = {}
    for name, estimator in get_candidates().items():
        print(f"  Training {name} on {track} train ONLY ...", flush=True)
        estimator.fit(X_train, y_train)
        probs = estimator.predict_proba(X_val)[:, 1]
        threshold, curve = _best_threshold(y_val, probs)
        score = _metrics(y_val, probs, threshold)
        candidate_results.append({"model": name, **score})
        validation_probs[name] = probs
        threshold_curves[name] = curve
        fitted_models[name] = estimator
        # For future inference on raw-feature tables: validator + Stage 2 preprocessor + model.
        fitted_pipeline = Pipeline([
            ("input_adapter", TrackInputAdapter(track, features)),
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        joblib.dump(fitted_pipeline, MODELS / f"{track}_{name}.joblib")
        print(f"    Validation AP={score['pr_auc_average_precision']:.4f}  ROC-AUC={score['roc_auc']:.4f}  "
              f"F1={score['f1']:.4f} at threshold={threshold:.2f}", flush=True)

    # Rank only on validation; never use the test split to choose algorithm.
    ranking = sorted(candidate_results,
                     key=lambda r: (-r["pr_auc_average_precision"], -r["roc_auc"], -r["f1"], r["model"]))
    winner = ranking[0]
    best_name = winner["model"]
    best_threshold = winner["threshold"]
    best_pipeline = Pipeline([
        ("input_adapter", TrackInputAdapter(track, features)),
        ("preprocessor", preprocessor),
        ("model", fitted_models[best_name]),
    ])
    # Verify the serialized inference contract before touching test data.
    replay = best_pipeline.predict_proba(data["val"][features])[:, 1]
    if not np.allclose(replay, validation_probs[best_name], atol=1e-10):
        raise AssertionError("Saved pipeline and validated prediction pipeline disagree")
    joblib.dump(best_pipeline, MODELS / f"{track}_best_model.pkl")

    sensitivity = None
    if track == "iranian":
        print("  Checking Status feature sensitivity on Iranian validation ...", flush=True)
        sensitivity = _status_sensitivity(best_name, data["train"], data["val"], meta, winner)
        (REPORTS / "iranian_status_sensitivity.json").write_text(
            json.dumps(sensitivity, indent=2, allow_nan=False), encoding="utf-8"
        )

    # FINAL: test is used only AFTER model, hyperparameters and threshold are locked.
    print(f"  Locked {best_name}; measuring ONE final test evaluation ...", flush=True)
    X_test_raw = data["test"][features]
    y_test = data["test"]["Churn"].to_numpy()
    test_probs = best_pipeline.predict_proba(X_test_raw)[:, 1]
    test_metrics = _metrics(y_test, test_probs, best_threshold)
    _save_candidate_chart(track, candidate_results)
    _save_threshold_chart(track, threshold_curves[best_name], best_threshold)
    _save_test_charts(track, y_test, test_probs, best_threshold)
    rows = [{"model": x["model"], "threshold": x["threshold"],
             **{k: x[k] for k in ("precision", "recall", "f1", "roc_auc",
                                "pr_auc_average_precision", "brier_score")}}
            for x in candidate_results]
    pd.DataFrame(rows).to_csv(REPORTS / f"{track}_candidate_validation_metrics.csv", index=False)
    output = {
        "track": track,
        "data_source": ("UCI Iranian churn (not Teletalk)" if track == "iranian"
                        else "IBM fictional public benchmark (not Teletalk)"),
        "training_rows": len(data["train"]),
        "validation_rows": len(data["val"]),
        "test_rows": len(data["test"]),
        "input_features": features,
        "transformed_feature_count": int(X_train.shape[1]),
        "raw_dataset_sha256_from_stage2": meta["raw_sha256"],
        "candidate_models": candidate_results,
        "selection_rule": "Highest validation average precision (AP/PR-AUC); tie-break ROC-AUC, F1",
        "threshold_rule": "Max validation F1 over 0.10..0.90 step 0.01; no economic costs assumed",
        "best_model_name": best_name,
        "selected_threshold": float(best_threshold),
        "best_model_file": f"models/{track}_best_model.pkl",
        "best_model_validation_metrics": winner,
        "best_model_final_test_metrics": test_metrics,
        "reliability": "Brier score + quantile-bin calibration curve; no post-hoc probability calibration applied",
        "status_sensitivity_validation_only": sensitivity,
        "limitations": [
            "Public benchmark/foreign operator performance is NOT Teletalk subscriber performance.",
            "Split and model/threshold selection use fixed random state 42; no external temporal validation.",
            "A validation-optimized F1 threshold does not reflect real-world intervention costs.",
        ],
    }
    (REPORTS / f"{track}_results.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    return output
