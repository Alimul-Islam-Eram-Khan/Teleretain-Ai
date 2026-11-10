"""TeleRetain AI Stage 5: strict source-aware routing to validated saved models.

predict_customer({"track": "iranian" | "ibm", "features": {...}})
returns a Pydantic PredictionOutput. Errors are raised, never silently corrected.

This is a public-dataset proof of concept, NOT a Teletalk production service.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import math

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from src.data_prep_common import ROOT
from src.schemas import TRACK_SCHEMAS, IranianInput, IBMInput, PredictionOutput

LABELS = {
    "iranian": "Iranian Track (UCI Iranian Churn)",
    "ibm": "IBM Benchmark (public fictional sample)",
}
EXPECTED_MODEL_NAMES = {"iranian": "random_forest", "ibm": "xgboost"}
DISCLAIMER = (
    "Proof of concept using a public non-Teletalk dataset. "
    "This is a model-estimated score, not a calibrated or causal claim, "
    "not a prediction for a real Teletalk subscriber."
)


def decide_track(request: dict[str, Any]) -> str:
    """Require an explicit, known track; never guess from ambiguous input."""
    if not isinstance(request, dict):
        raise ValueError("Request must be an object with 'track' and 'features'.")
    if set(request) != {"track", "features"}:
        raise ValueError("Request must contain ONLY 'track' and 'features' (both required).")
    track = request["track"]
    if not isinstance(track, str) or track not in TRACK_SCHEMAS:
        raise ValueError("Explicit track must be 'iranian' or 'ibm'; cannot infer track.")
    if not isinstance(request["features"], dict):
        raise ValueError("'features' must be an object of original named dataset features.")
    return track


def _read_report(track: str) -> dict:
    filename = ROOT / "reports" / "stage3" / f"{track}_results.json"
    if not filename.is_file():
        raise FileNotFoundError(f"Stage 3 results missing: {filename}")
    report = json.loads(filename.read_text(encoding="utf-8"))
    if report.get("track") != track:
        raise ValueError("Stage 3 report is tagged for a different track.")
    if report.get("best_model_name") != EXPECTED_MODEL_NAMES[track]:
        raise ValueError(
            f"This Stage 5 release expects {track}={EXPECTED_MODEL_NAMES[track]}; "
            "do not silently swap trained models."
        )
    return report


@lru_cache(maxsize=2)
def _get_context(track: str) -> tuple[Pipeline, dict, dict[str, set[str]]]:
    """Load ONLY locally trained, trusted files. Never deserialize user uploads."""
    if track not in TRACK_SCHEMAS:
        raise ValueError("Unknown track")
    report = _read_report(track)
    filename = ROOT / "models" / f"{track}_best_model.pkl"
    if not filename.is_file():
        raise FileNotFoundError(f"Trained model missing: {filename}")
    pipeline = joblib.load(filename)  # Trusted local Stage 3 training artifact ONLY.
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != [
        "input_adapter", "preprocessor", "model"
    ]:
        raise ValueError("Saved pipeline structure mismatch.")
    adapter = pipeline.named_steps["input_adapter"]
    schema_features = list(TRACK_SCHEMAS[track].model_fields.keys())
    schema_aliases = {
        f.alias or name for name, f in TRACK_SCHEMAS[track].model_fields.items()
    }
    expected_features = report.get("input_features", [])
    if (
        getattr(adapter, "track", None) != track
        or list(getattr(adapter, "feature_columns", [])) != list(expected_features)
        or schema_aliases != set(expected_features)
        or len(schema_features) != len(expected_features)
    ):
        raise ValueError("Saved model, source report and Pydantic schema disagree.")
    threshold = float(report["selected_threshold"])
    if not (math.isfinite(threshold) and 0 < threshold < 1):
        raise ValueError("Invalid locked decision threshold in Stage 3 report.")
    model = pipeline.named_steps["model"]
    class_name = type(model).__name__
    if class_name != {"iranian": "RandomForestClassifier", "ibm": "XGBClassifier"}[track]:
        raise ValueError(f"Unexpected {track} model type: {class_name}")
    meta_path = ROOT / "reports" / "stage2" / f"{track}_data_quality.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 data-quality report: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("track") != track or metadata.get("feature_columns") != expected_features:
        raise ValueError("Stage 2 feature metadata does not match saved model.")
    categories = metadata["categorical_columns"]
    preprocessing = pipeline.named_steps["preprocessor"]
    encoder = preprocessing.named_transformers_["categorical"].named_steps["onehot"]
    if len(categories) != len(encoder.categories_):
        raise ValueError("Categorical encoder dimensions do not match Stage 2 report.")
    allowed = {
        name: {str(value) for value in values}
        for name, values in zip(categories, encoder.categories_)
    }
    return pipeline, report, allowed


def _checked_features(track: str, features: dict[str, Any]) -> dict:
    model_cls = TRACK_SCHEMAS[track]
    parsed: IranianInput | IBMInput = model_cls.model_validate(features)
    # Canonical names follow the exact Stage 2 feature contract.
    parsed_features = parsed.model_dump(by_alias=True)
    _, _, allowed = _get_context(track)
    for name, choices in allowed.items():
        value = parsed_features[name]
        if str(value) not in choices:
            raise ValueError(
                f"Invalid {track} category for {name!r}: {value!r}. "
                f"Training categories: {sorted(choices)}"
            )
    return parsed_features


def predict_customer(request: dict[str, Any]) -> PredictionOutput:
    """One validated real PUBLIC-DATA-shaped input row -> source-labelled score.

    Pydantic ValidationError means missing/wrong values; ValueError means
    bad routing, unknown categories, or model contract mismatches.
    """
    track = decide_track(request)
    cleaned = _checked_features(track, request["features"])
    pipeline, report, _ = _get_context(track)
    ordered = {column: cleaned[column] for column in report["input_features"]}
    row = pd.DataFrame([ordered])
    probability = float(pipeline.predict_proba(row)[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid model churn score; no prediction returned.")
    threshold = float(report["selected_threshold"])
    cls = int(probability >= threshold)
    return PredictionOutput(
        track=track,
        data_source_label=LABELS[track],
        data_source_description=report["data_source"],
        selected_model=report["best_model_name"],
        churn_probability=probability,
        decision_threshold=threshold,
        predicted_class=cls,
        predicted_label="Churn" if cls == 1 else "No churn",
        disclaimer=DISCLAIMER,
    )


def missing_fields_for(track: str, features: dict[str, Any]) -> list[str]:
    """For the future chatbot: tell the user which fields are missing, not guesses."""
    if track not in TRACK_SCHEMAS or not isinstance(features, dict):
        raise ValueError("Choose a valid track and feature object first.")
    schema = TRACK_SCHEMAS[track]
    required = {field.alias or name for name, field in schema.model_fields.items() if field.is_required()}
    return sorted(required - set(features))
