"""Read-only checks on real, locally processed data — no generated customer records."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd


def check(track):
    quality = ROOT / "reports" / "stage2" / f"{track}_data_quality.json"
    assert quality.is_file(), f"Missing {quality}"
    meta = json.loads(quality.read_text(encoding="utf-8"))
    parts = {}
    for split in ("train", "val", "test"):
        filename = ROOT / "data" / "processed" / track / f"{split}.csv"
        assert filename.is_file(), f"Missing {filename}"
        parts[split] = pd.read_csv(filename)
        assert len(parts[split]) == meta["splits"][split]["rows"]
        assert set(parts[split]["Churn"].dropna().unique()).issubset({0, 1})
    assert sum(map(len, parts.values())) == meta["cleaned_rows"]
    keys = ["customerID"] if track == "ibm" else meta["feature_columns"]
    hashes = [set(pd.util.hash_pandas_object(parts[s][keys], index=False).tolist())
              for s in ("train", "val", "test")]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2]), "Potential cross-split overlap"
    if track == "iranian":
        assert meta["checks"]["same_features_conflicting_label_groups"] >= 0
        assert "StratifiedGroupKFold" in meta["split_method"]
    modelpath = ROOT / "models" / f"{track}_preprocessor.joblib"
    assert modelpath.is_file(), f"Missing {modelpath}"
    pre = joblib.load(modelpath)
    features = meta["feature_columns"]
    # Read-only transforms; no fit/refit on validation or test.
    for split in ("train", "val", "test"):
        transformed = pre.transform(parts[split][features])
        assert len(transformed) == len(parts[split])
    print(f"PASS: {track} | rows={meta['cleaned_rows']} | model-input features={len(features)} | split isolation OK")


def main():
    check("iranian")
    check("ibm")
    print("SUCCESS: Stage 2 outputs validated; no ML models have been trained yet.")


if __name__ == "__main__":
    main()
