"""Iranian real-operator dataset: separate preprocessing and split."""
from pathlib import Path
import pandas as pd
from src.data_prep_common import ROOT, save_outputs

RAW = ROOT / "data" / "raw" / "iranian" / "Customer Churn.csv"
FEATURES = [
    "Call Failure", "Complains", "Subscription Length", "Charge Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status",
    "Age", "Customer Value",
]
CATEGORICAL = ["Complains", "Tariff Plan", "Status"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def run() -> dict:
    if not RAW.is_file():
        raise FileNotFoundError(f"Iranian CSV missing: {RAW}")
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    # Fix UCI CSV column names with double spaces (not feature meanings).
    df.columns = [" ".join(c.split()) for c in df.columns]
    expected = set(FEATURES + ["Churn"])
    if set(df.columns) != expected:
        raise ValueError(f"Iranian columns mismatch. Missing: {expected - set(df.columns)}. Extra: {set(df.columns) - expected}")
    for col in FEATURES + ["Churn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    nulls = {key: int(value) for key, value in df.isna().sum().items() if value > 0}
    if nulls:
        raise ValueError(f"Unexpected missing/invalid Iranian data: {nulls}")
    if not set(df["Churn"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected Iranian churn target; only 0 and 1 allowed.")
    df["Churn"] = df["Churn"].astype(int)

    before = len(df)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    # Conflicting outcomes do not justify altering or discarding churn labels.
    # Since no customerID is available, keep *all* identical feature vectors
    # (including mixed-label groups) within a single data split.
    distinct_labels_per_vector = df.groupby(FEATURES, dropna=False)["Churn"].nunique()
    conflict_count = int((distinct_labels_per_vector > 1).sum())
    conflicting_rows = int((df.groupby(FEATURES, dropna=False)["Churn"]
                            .transform("nunique") > 1).sum())
    checks = {
        "original_rows": before,
        "exact_duplicates_removed_before_splitting": duplicates,
        "same_features_conflicting_label_groups": conflict_count,
        "rows_in_conflicting_groups_after_exact_deduplication": conflicting_rows,
        "conflict_handling": "Preserve both labels; group by complete input feature vector during train/val/test split",
        "missing_after_numeric_conversion": 0,
        "source": "UCI Iranian Churn; NOT Teletalk customer data",
        "status_caution": "Status (active/non-active) will receive leakage-sensitivity analysis during model evaluation."
    }
    return save_outputs("iranian", df, NUMERIC, CATEGORICAL,
                        key_cols=FEATURES, raw_path=RAW, checks=checks,
                        group_feature_cols=FEATURES)
