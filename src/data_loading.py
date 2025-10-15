# src/data_loading.py

from pathlib import Path
import pandas as pd
import numpy as np

def load_raw_listings(path: Path) -> pd.DataFrame:
    """
    Load InsideAirbnb listings CSV with robust handling for:
      - Semicolons vs commas
      - UTF-8/BOM issues
      - Missing value normalization
      - Price cleanup when present
    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"List of listings not found at: {path}")

    # Try common encodings
    encodings = ["utf-8", "latin-1"]
    seps = [",", ";"]

    last_err = None
    df = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                # Heuristic: if we got only 1 column after reading, wrong sep — skip
                if df.shape[1] == 1:
                    continue
                break
            except Exception as e:
                last_err = e
        if df is not None and df.shape[1] > 1:
            break

    if df is None or df.shape[1] == 1:
        raise ValueError(
            f"Failed to parse CSV with common encodings/separators. Last error: {last_err}"
        )

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Standardize price column if present
    if "price" in df.columns:
        s = df["price"].astype(str).str.replace(r"[^\d,.\-]", "", regex=True).str.replace(" ", "")
        # Handle different decimal formats (e.g., "1.234,56" vs "1,234.56")
        def to_float(x):
            if x == "" or x is None:
                return np.nan
            if x.count(",") == 1 and x.count(".") >= 1 and x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
            try:
                return float(x)
            except:
                return np.nan
        df["price"] = s.apply(to_float)

    # (Optional) map common boolean-like columns
    for col in ["host_is_superhost", "instant_bookable"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map({"t": 1, "true": 1, "yes": 1, "f": 0, "false": 0, "no": 0})
                .fillna(0)
                .astype(int)
            )

    return df