# src/train_price_model.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    XGB_AVAILABLE = False

from src.data_loading import load_raw_listings

# Primary import
try:
    from src.feature_engineering import basic_feature_engineering
except Exception as e:
    print("[WARN] Could not import basic_feature_engineering from src.feature_engineering:", e)
    # Fallback inline implementation
    def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "price" in df.columns:
            # simple cleaner
            s = df["price"].astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
            s = s.str.replace(",", "").str.replace(" ", "")
            df["price"] = pd.to_numeric(s, errors="coerce")
        # Bathrooms fallback
        if "bathrooms" not in df.columns and "bathrooms_text" in df.columns:
            df["bathrooms"] = (
                df["bathrooms_text"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)").astype(float)
            )
        # Select typical modeling features
        cols = [c for c in [
            "accommodates", "bedrooms", "bathrooms", "beds",
            "minimum_nights", "maximum_nights", "room_type",
            "property_type", "host_is_superhost", "availability_365"
        ] if c in df.columns]
        out = df[cols].copy()
        if "price" in df.columns:
            out["price"] = df["price"]
        # fill numeric
        for col in out.select_dtypes(include=[np.number]).columns:
            out[col] = out[col].fillna(out[col].median())
        return out

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "listings.csv"
API_DIR = PROJECT_ROOT / "api"
API_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "listings_processed.csv"

MODEL_PATH = API_DIR / "model_price.pkl"
PREPROC_PATH = API_DIR / "preprocessor_price.pkl"

def main():
    print(f"Loading data: {DATA_RAW}")
    df = load_raw_listings(DATA_RAW)
    print(f"Raw shape: {df.shape}")

    df_fe = basic_feature_engineering(df)
    print(f"Engineered shape: {df_fe.shape}")

    if "price" not in df_fe.columns:
        raise ValueError("Target 'price' missing after feature engineering.")

    df_fe = df_fe.dropna(subset=["price"])
    df_fe = df_fe[df_fe["price"] > 0]
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_fe.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed dataset: {PROCESSED_PATH}")

    y = df_fe["price"].values
    X = df_fe.drop(columns=["price"])

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4
        )
        print("Using XGBRegressor")
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        print("Using RandomForestRegressor")

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training price model...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"MAE: {mae:.2f}, R²: {r2:.3f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model pipeline: {MODEL_PATH}")

    joblib.dump({
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "all_columns": num_cols + cat_cols,
    }, PREPROC_PATH)
    print(f"Saved preprocessor info: {PREPROC_PATH}")

if __name__ == "__main__":
    main()