# src/feature_engineering.py

import pandas as pd
import numpy as np

def clean_price_series(s: pd.Series) -> pd.Series:
    """
    Clean price strings like "$1,234.56", "€1.234,56", "1 234,56 €", etc. -> numeric.
    """
    s = s.astype(str).str.replace(r"[^\d,.\-]", "", regex=True).str.replace(" ", "", regex=False)

    def to_float(x: str):
        if x == "" or x is None:
            return np.nan
        # If decimal is comma and thousand is dot: "1.234,56" -> "1234.56"
        if x.count(",") == 1 and x.count(".") >= 1 and x.rfind(",") > x.rfind("."):
            x = x.replace(".", "").replace(",", ".")
        else:
            # Otherwise remove thousands separators
            if x.count(",") > 1 and "." not in x:
                x = x.replace(",", "")
            x = x.replace(",", "")
        try:
            return float(x)
        except:
            return np.nan

    return s.apply(to_float)

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineering common Airbnb features for price and availability modeling.
    Handles slightly inconsistent column names and missing values.
    """
    df = df.copy()

    # Price handling
    if "price" in df.columns:
        df["price"] = clean_price_series(df["price"])

    # Bathrooms numeric from 'bathrooms_text' fallback
    if "bathrooms" not in df.columns and "bathrooms_text" in df.columns:
        df["bathrooms"] = (
            df["bathrooms_text"]
            .astype(str)
            .str.extract(r"([0-9]*\.?[0-9]+)")
            .astype(float)
        )

    # Neighbourhood feature: choose best available
    if "neighbourhood_cleansed" in df.columns:
        neighbourhood_col = "neighbourhood_cleansed"
    elif "neighbourhood" in df.columns:
        neighbourhood_col = "neighbourhood"
    else:
        neighbourhood_col = None

    # Common features across InsideAirbnb datasets
    feature_candidates = [
        "accommodates",
        "bedrooms",
        "bathrooms",
        "beds",
        "minimum_nights",
        "maximum_nights",
        "room_type",
        "property_type",
        "host_is_superhost",
        "latitude",
        "longitude",
        "availability_365",
        "minimum_nights_avg_ntm",
    ]
    if neighbourhood_col:
        feature_candidates.append(neighbourhood_col)

    present = [c for c in feature_candidates if c in df.columns]
    df_sub = df[present].copy()

    # Coerce booleans
    if "host_is_superhost" in df_sub.columns:
        df_sub["host_is_superhost"] = (
            df_sub["host_is_superhost"]
            .astype(str)
            .str.lower()
            .map({"t": 1, "true": 1, "f": 0, "false": 0})
            .fillna(0)
            .astype(int)
        )

    # Fill numeric NaNs
    for col in df_sub.select_dtypes(include=[np.number]).columns:
        df_sub[col] = df_sub[col].fillna(df_sub[col].median())

    # Add target 'price' if available
    if "price" in df.columns:
        df_sub["price"] = df["price"]

    return df_sub