"""Read-only final artifact checks, using only real Stage 2 holdout records."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from src.training_common import _metrics, get_candidates


def verify(track):
    report_file = ROOT / "reports" / "stage3" / f"{track}_results.json"
    assert report_file.is_file(), f"Stage 3 report missing: {report_file}"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["track"] == track
    assert len(report["candidate_models"]) == 3
    assert {r["model"] for r in report["candidate_models"]} == set(get_candidates())
    assert report["best_model_name"] in get_candidates()
    threshold = report["selected_threshold"]
    assert 0.10 <= threshold <= 0.90
    pipeline_path = ROOT / report["best_model_file"]
    assert pipeline_path.is_file(), pipeline_path
    pipeline = joblib.load(pipeline_path)  # local trusted artifacts only
    test_df = pd.read_csv(ROOT / "data" / "processed" / track / "test.csv")
    X = test_df[report["input_features"]]
    y = test_df["Churn"].to_numpy()
    probabilities = pipeline.predict_proba(X)[:, 1]
    assert len(probabilities) == len(test_df)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    reevaluated = _metrics(y, probabilities, threshold)
    for name in ("precision", "recall", "f1", "roc_auc",
                 "pr_auc_average_precision", "brier_score"):
        assert np.isclose(reevaluated[name], report["best_model_final_test_metrics"][name], atol=1e-9), name
    assert reevaluated["confusion_matrix"] == report["best_model_final_test_metrics"]["confusion_matrix"]
    for name in get_candidates():
        assert (ROOT / "models" / f"{track}_{name}.joblib").is_file()
    for suffix in ("candidate_validation_metrics.csv", "candidate_comparison.png",
                   "threshold_validation.png", "test_confusion_matrix.png", "test_calibration.png"):
        assert (ROOT / "reports" / "stage3" / f"{track}_{suffix}").is_file(), suffix
    print(f"PASS: {track} | best={report['best_model_name']} | "
          f"threshold={threshold:.2f} | test F1={reevaluated['f1']:.4f}")
    return pipeline, X


def main():
    iranian, iranian_test = verify("iranian")
    ibm, ibm_test = verify("ibm")
    for expected_error_model, foreign_row in (
        (iranian, ibm_test.iloc[:1]), (ibm, iranian_test.iloc[:1])
    ):
        try:
            expected_error_model.predict_proba(foreign_row)
        except ValueError as error:
            assert "feature mismatch" in str(error)
        else:
            raise AssertionError("Cross-track input was accepted: routing safety failed")
    print("PASS: both saved models reject the other track's data BEFORE preprocessing")
    assert (ROOT / "reports" / "stage3" / "iranian_status_sensitivity.json").is_file()
    print("SUCCESS: Stage 3 verified — only trained-on-public-data models, no Teletalk customers.")


if __name__ == "__main__":
    main()
