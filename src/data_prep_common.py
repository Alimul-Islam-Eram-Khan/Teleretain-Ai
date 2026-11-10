"""Common, track-agnostic utilities. Never combine Iranian and IBM data."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "stage2"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop", verbose_feature_names_out=True)


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["Churn"].value_counts().sort_index().items()}


def _verify_disjoint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                     key_cols: list[str]) -> None:
    def keys(data: pd.DataFrame) -> set[int]:
        return set(pd.util.hash_pandas_object(data[key_cols], index=False).astype("uint64").tolist())
    t, v, s = keys(train), keys(val), keys(test)
    if t & v or t & s or v & s:
        raise ValueError("Data leakage danger: identical records/keys cross train, val and test splits.")


def stratified_split(df: pd.DataFrame, *, key_cols: list[str],
                     group_feature_cols: list[str] | None = None):
    """Separate 70/15/15 holdouts without putting equivalent inputs in multiple splits.

    Iranian: all rows sharing the 13-feature vector form ONE indivisible group.
    They may genuinely have different churn labels; neither label is changed.
    IBM: customerID is unique, so conventional stratified splitting is fine.
    """
    if group_feature_cols is not None:
        # Factorize complete feature tuples; Churn is intentionally NOT part of a group.
        groups = pd.factorize(pd.MultiIndex.from_frame(df[group_feature_cols]),
                              sort=False)[0]
        splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
        folds = [test_ix for _, test_ix in splitter.split(
            df[group_feature_cols], df["Churn"], groups=groups
        )]
        import numpy as np
        train_ix = np.concatenate(folds[:14])  # approx. 70% of samples
        val_ix = np.concatenate(folds[14:17])  # approx. 15%
        test_ix = np.concatenate(folds[17:])   # approx. 15%
        train, val, test = (df.iloc[ix].reset_index(drop=True)
                            for ix in (train_ix, val_ix, test_ix))
    else:
        train, rest = train_test_split(
            df, train_size=0.70, random_state=42, stratify=df["Churn"]
        )
        val, test = train_test_split(
            rest, train_size=0.50, random_state=42, stratify=rest["Churn"]
        )
        train, val, test = (part.reset_index(drop=True) for part in (train, val, test))

    _verify_disjoint(train, val, test, key_cols)
    for name, split in (("train", train), ("validation", val), ("test", test)):
        if split["Churn"].nunique() != 2:
            raise ValueError(f"{name} split is missing one churn class; review grouping.")
    return train, val, test


def make_charts(track: str, df: pd.DataFrame, numeric_cols: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts = df["Churn"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Non-churn (0)", "Churn (1)"], counts.values)
    ax.bar_label(bars)
    ax.set_title(f"{track.title()} track: churn distribution (training only)")
    ax.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"{track}_churn_distribution.png", dpi=160)
    plt.close(fig)

    for col in numeric_cols[:3]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=24)
        ax.set_title(f"{track.title()} track: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Records")
        fig.tight_layout()
        filename = col.lower().replace(" ", "_")
        fig.savefig(REPORT_DIR / f"{track}_{filename}_hist.png", dpi=160)
        plt.close(fig)


def save_outputs(track: str, df: pd.DataFrame, numeric: list[str],
                 categorical: list[str], *, key_cols: list[str],
                 raw_path: Path, checks: dict,
                 group_feature_cols: list[str] | None = None) -> dict:
    train, val, test = stratified_split(
        df, key_cols=key_cols, group_feature_cols=group_feature_cols
    )
    target_dir = PROCESSED_DIR / track
    target_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(target_dir / f"{name}.csv", index=False)

    preprocessor = make_preprocessor(numeric, categorical)
    feature_cols = numeric + categorical
    # IMPORTANT: fit ONLY on train! Do not fit on val/test.
    preprocessor.fit(train[feature_cols])
    for name, part in (("train", train), ("val", val), ("test", test)):
        result = preprocessor.transform(part[feature_cols])
        if result.shape[0] != len(part):
            raise ValueError(f"Unexpected transformed row count for {name}")

    joblib.dump(preprocessor, MODELS_DIR / f"{track}_preprocessor.joblib")
    make_charts(track, train, numeric)
    result = {
        "track": track,
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_sha256": file_sha256(raw_path),
        "cleaned_rows": len(df),
        "cleaned_columns": len(df.columns),
        "class_distribution_cleaned": _label_counts(df),
        "splits": {
            name: {"rows": len(part), "class_distribution": _label_counts(part)}
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "feature_columns": feature_cols,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "transformed_feature_count": len(preprocessor.get_feature_names_out()),
        "target": "Churn (0=non-churn, 1=churn)",
        "preprocessor_fitted_on": "train only",
        "random_state": 42,
        "split_method": (
            "StratifiedGroupKFold 20 folds: 14 train, 3 validation, 3 test; "
            "grouped by all input features" if group_feature_cols is not None
            else "Two-stage stratified train_test_split"
        ),
        "checks": checks,
    }
    (REPORT_DIR / f"{track}_data_quality.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
