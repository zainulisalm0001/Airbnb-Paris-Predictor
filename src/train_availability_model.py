# src/train_availability_model.py

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data_loading import load_raw_listings
from .feature_engineering import basic_feature_engineering


def choose_target(df: pd.DataFrame) -> pd.Series:
    """
    For availability prediction:
    - If 'availability_30' exists, define 'low_avail_next_month' as 1 if < 10 days available.
    - Else if 'availability_365' exists, define 'low_avail_year' as 1 if < 180 days available.
    - Otherwise raise error.

    Returns: y (Series with 0/1), and string description of the target.
    """
    if "availability_30" in df.columns:
        y = (df["availability_30"] < 10).astype(int)
        target_desc = "low_availability_30 (< 10 days available next 30 days)"
        return y, target_desc
    elif "availability_365" in df.columns:
        y = (df["availability_365"] < 180).astype(int)
        target_desc = "low_availability_365 (< 180 days available)"
        return y, target_desc
    else:
        raise ValueError(
            "No suitable availability target found. Need 'availability_30' or 'availability_365'."
        )


def main():
    ROOT = Path(__file__).resolve().parents[1]
    data_path = ROOT / "data" / "raw" / "listings.csv"
    out_model = ROOT / "api" / "model_availability.pkl"
    out_preproc = ROOT / "api" / "preprocessor_availability.pkl"
    out_proc = ROOT / "data" / "processed" / "listings_processed.csv"

    print(f"Loading data: {data_path}")
    df_raw = load_raw_listings(data_path)
    print(f"Raw shape: {df_raw.shape}")

    # Engineer features
    df_feat = basic_feature_engineering(df_raw)
    print(f"Engineered shape: {df_feat.shape}")

    # Save processed dataset
    df_feat.to_csv(out_proc, index=False)
    print(f"Saved processed dataset: {out_proc}")

    # Choose features — price is not required for availability, so drop it if it exists
    feature_cols = [c for c in df_feat.columns if c not in ["price"]]
    if len(feature_cols) < 5:
        raise ValueError("Not enough features found for availability model.")

    X = df_feat[feature_cols].copy()

    # Target definition
    y, target_desc = choose_target(df_raw)
    print(f"Target: {target_desc} — positive rate: {y.mean():.2f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessor: scale numeric, OHE categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Model: XGBClassifier
    clf = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    print("Training availability model...")
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")

    # Save model
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_model)
    print(f"Saved model pipeline: {out_model}")

    # Save preprocessor info (columns)
    preproc_info = {
        "columns": feature_cols
    }
    with open(out_preproc, "w") as f:
        json.dump(preproc_info, f, indent=2)
    print(f"Saved preprocessor info: {out_preproc}")

if __name__ == "__main__":
    main()