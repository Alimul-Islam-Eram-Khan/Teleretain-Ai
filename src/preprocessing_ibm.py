"""IBM public benchmark: entirely separate preprocessing and split."""
import pandas as pd
from src.data_prep_common import ROOT, save_outputs

RAW = ROOT / "data" / "raw" / "ibm" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"
ID = "customerID"


def run() -> dict:
    if not RAW.is_file():
        raise FileNotFoundError(f"IBM CSV missing: {RAW}")
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    required = set(NUMERIC + [TARGET, ID])
    if not required.issubset(df.columns):
        raise ValueError(f"IBM columns missing: {required - set(df.columns)}")
    if df[ID].isna().any() or df[ID].duplicated().any():
        raise ValueError("IBM customerID is missing or duplicated. Stop for review.")
    before = len(df)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        raise ValueError(f"IBM contains {duplicate_rows} unexpected exact duplicate records.")

    # Blank strings are not counted by initial df.isna(); coerce them to NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
        errors="coerce",
    )
    for col in ["tenure", "MonthlyCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    unrecognized = set(df[TARGET].dropna().unique()) - {"Yes", "No"}
    if unrecognized or df[TARGET].isna().any():
        raise ValueError(f"Unexpected IBM target labels: {unrecognized}")
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    categorical = [c for c in df.columns if c not in NUMERIC + [TARGET, ID]]
    # SeniorCitizen is a categorical indicator rather than continuous quantity.
    if "SeniorCitizen" in categorical:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype("string")
    missing_numeric = {c: int(df[c].isna().sum()) for c in NUMERIC}
    bad_categorical = {c: int(df[c].isna().sum()) for c in categorical if df[c].isna().any()}
    if bad_categorical or missing_numeric["tenure"] or missing_numeric["MonthlyCharges"]:
        raise ValueError(f"Unexpected missing IBM values: numeric={missing_numeric}, categorical={bad_categorical}")
    checks = {
        "original_rows": before,
        "original_exact_duplicate_rows": duplicate_rows,
        "distinct_customer_ids": int(df[ID].nunique()),
        "totalcharges_missing_after_conversion": missing_numeric["TotalCharges"],
        "totalcharges_missing_strategy": "Train-only median imputation; preserve rows and missingness in split CSVs",
        "customerID": "retained for split audit; excluded from model features",
        "source": "IBM public fictional benchmark; NOT Teletalk customer data",
    }
    return save_outputs("ibm", df, NUMERIC, categorical,
                        key_cols=[ID], raw_path=RAW, checks=checks)
