"""Read-only SHAP/report integrity and track isolation verification."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.explain import load_track, explain_one

OUT = ROOT / "reports" / "stage4"


def verify_one(track):
    ctx = load_track(track)
    report_file = OUT / f"{track}_summary.json"
    if not report_file.is_file():
        raise FileNotFoundError(f"Stage 4 output missing: {report_file}")
    result = json.loads(report_file.read_text(encoding="utf-8"))
    assert result["track"] == track and result["best_model"] == ctx.report["best_model_name"]
    assert result["classification_threshold_locked_from_stage3"] == ctx.report["selected_threshold"]
    assert len(result["global_mean_absolute_contributions"]) == len(ctx.feature_names)
    assert {r["feature"] for r in result["global_mean_absolute_contributions"]} == set(ctx.feature_names)
    assert result["sample_size"] >= 2
    assert result["dataset_split_explained"].startswith("validation only")
    for suffix in ("global_importance.png", "beeswarm.png"):
        filename = OUT / f"{track}_{suffix}"
        assert filename.is_file() and filename.stat().st_size > 1000, filename
    validation = pd.read_csv(ROOT / "data" / "processed" / track / "val.csv")
    features = validation.loc[:, ctx.report["input_features"]]
    assert result["local_examples"], "No local examples were produced"
    for label, local_filename in result["local_examples"].items():
        local = json.loads((OUT / local_filename).read_text(encoding="utf-8"))
        index = local["validation_row_index"]
        assert 0 <= index < len(features)
        actual = explain_one(track, features.iloc[[index]])
        assert np.isclose(actual["probability_churn"], local["probability_churn"], atol=5e-6)
        assert actual["predicted_class"] == local["predicted_class"]
        assert actual["feature_contributions"] == local["feature_contributions"] or all(
            np.isclose(x["shap_value"], y["shap_value"], atol=1e-5)
            and x["feature"] == y["feature"]
            for x, y in zip(actual["feature_contributions"], local["feature_contributions"])
        )
        image_file = OUT / f"{track}_local_{label}_waterfall.png"
        assert image_file.is_file() and image_file.stat().st_size > 1000, image_file
    print(f"PASS: {track} | global + local explanation artifacts | "
          "prediction/additivity consistency")
    return features


def main():
    iranian = verify_one("iranian")
    ibm = verify_one("ibm")
    for track, wrong in (("iranian", ibm), ("ibm", iranian)):
        try:
            explain_one(track, wrong.iloc[:1])
        except ValueError as err:
            assert "feature mismatch" in str(err).lower(), str(err)
        else:
            raise AssertionError(f"FAILED: {track} accepted another track's feature row")
    print("PASS: Iranian and IBM raw feature contracts remain isolated")
    print("SUCCESS: Stage 4 SHAP explanations verified on saved models (validation only).")


if __name__ == "__main__":
    main()
