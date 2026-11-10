"""Dataframe contract for Stage-3-trained models.

In particular, Stage 2 fitted the IBM SeniorCitizen categories as strings
('0' and '1'). Reading the saved CSVs re-infers this column as integers,
so restore the fitting-time dtype before any preprocessing or inference.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TrackInputAdapter(BaseEstimator, TransformerMixin):
    """Reject cross-track/extra/missing columns and preserve feature dtypes."""

    def __init__(self, track: str, feature_columns: list[str]):
        self.track = track
        self.feature_columns = feature_columns

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Prediction input must be a pandas DataFrame with named columns.")
        received = set(X.columns)
        required = set(self.feature_columns)
        missing = sorted(required - received)
        unexpected = sorted(received - required)
        if missing or unexpected:
            raise ValueError(
                f"{self.track.upper()} track feature mismatch; "
                f"missing={missing}; unexpected={unexpected}. "
                "Do not mix Iranian and IBM model inputs."
            )
        X = X.loc[:, self.feature_columns].copy()
        if self.track == "ibm":
            # Matches the exact type used in Stage 2 before its fit.
            X["SeniorCitizen"] = X["SeniorCitizen"].astype("string")
        return X
