"""Read-only Stage 3 prerequisites check, BEFORE any model training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training_common import _load_track
from src.model_inputs import TrackInputAdapter


def main():
    print("Stage 3 preflight: reading Stage 2 public-data artifacts only")
    for track in ("iranian", "ibm"):
        meta, data, pre = _load_track(track)
        x_train = data["train"][meta["feature_columns"]]
        x_val = data["val"][meta["feature_columns"]]
        adapter = TrackInputAdapter(track, meta["feature_columns"])
        train = pre.transform(adapter.transform(x_train))
        val = pre.transform(adapter.transform(x_val))
        assert train.shape[1] == val.shape[1] == meta["transformed_feature_count"]
        if track == "ibm":
            encoder = pre.named_transformers_["categorical"].named_steps["onehot"]
            col_idx = meta["categorical_columns"].index("SeniorCitizen")
            assert {"0", "1"}.issubset({str(x) for x in encoder.categories_[col_idx]})
        print(f"PASS: {track} | train={train.shape} | validation={val.shape} | "
              "train-fitted preprocessor usable")
    print("SUCCESS: ready for python scripts/run_stage3.py")


if __name__ == "__main__":
    main()
