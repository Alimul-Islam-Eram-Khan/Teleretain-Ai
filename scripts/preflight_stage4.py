"""Read-only Stage 4 file/model compatibility check; never retrains a model."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import shap
import xgboost
from src.explain import load_track


def main():
    print(f"Installed shap={shap.__version__}, xgboost={xgboost.__version__}")
    for track in ("iranian", "ibm"):
        ctx = load_track(track)
        file = ROOT / "data" / "processed" / track / "val.csv"
        if not file.is_file():
            raise FileNotFoundError(f"Validation split missing: {file}")
        raw = pd.read_csv(file).loc[:, ctx.report["input_features"]].iloc[:1]
        prediction = ctx.pipeline.predict_proba(raw)[0, 1]
        print(f"PASS: {track} | {ctx.report['best_model_name']} | "
              f"{len(ctx.feature_names)} transformed features | "
              f"validation-row churn probability={prediction:.6f}")
    print("SUCCESS: Stage 4 preflight checks passed.")


if __name__ == "__main__":
    main()
