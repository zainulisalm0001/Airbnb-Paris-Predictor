# streamlit_app/pages/2_Availability_Predictor.py

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="📅 Availability Predictor", page_icon="📅", layout="centered")

# ---------- Styles ----------
st.markdown("""
<style>
.card {
    border-radius: 14px;
    padding: 18px 20px;
    background: #ffffff;
    border: 1px solid #eaeaef;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.04);
    margin: 0.75rem 0;
}
.pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #f5fff3;
    border: 1px solid #e0ffe1;
    color: #227733;
    font-weight: 600;
    font-size: 12px;
    margin-bottom: 10px;
}
.big-number {
    font-size: 34px; font-weight: 700; margin-top: 0.25rem;
}
.info-note {
    color: #666; font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="pill">AVAILABILITY PREDICTION</div>', unsafe_allow_html=True)
st.title("📅 Availability Insight")
st.write(
    "Estimate how many days your listing may be available over a year — to plan pricing and hosting frequency."
)

# ---------- Paths ----------
def find_root() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parents[2], here.parents[1], Path.cwd()]:
        if (base / "api").exists():
            return base
    return here.parents[2]

ROOT = find_root()
MODEL_PATH = ROOT / "api" / "model_availability.pkl"
PREPROC_PATH = ROOT / "api" / "preprocessor_availability.pkl"  # optional

@st.cache_resource
def load_availability_pipeline():
    if not MODEL_PATH.exists():
        return None, f"Availability model not found at: `{MODEL_PATH}`. Train it using `python -m src.train_availability_model`."
    try:
        pipe = joblib.load(MODEL_PATH)
        return pipe, ""
    except Exception as e:
        return None, f"Failed to load availability model: {e}"

@st.cache_resource
def load_preproc_columns():
    if PREPROC_PATH.exists():
        try:
            obj = joblib.load(PREPROC_PATH)
            if isinstance(obj, dict) and "columns" in obj:
                return obj["columns"]
        except Exception:
            pass
    # Fallback
    return [
        "accommodates", "bedrooms", "bathrooms", "beds",
        "minimum_nights", "maximum_nights",
        "room_type", "property_type", "host_is_superhost",
        "latitude", "longitude",
        "availability_365", "minimum_nights_avg_ntm",
        "neighbourhood_cleansed",
    ]

pipe, err = load_availability_pipeline()
if pipe is None:
    st.error(err)
    st.stop()

expected_cols = load_preproc_columns()

# ---------- Inputs ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Tell us about your place")

col1, col2 = st.columns(2)
with col1:
    accommodates = st.number_input("How many guests can you host?", 1, 16, 2)
    bedrooms = st.number_input("How many bedrooms?", 0, 10, 1)
    bathrooms = st.number_input("How many bathrooms?", 0.0, 10.0, 1.0, step=0.5)
    beds = st.number_input("How many beds?", 0, 12, 1)
with col2:
    min_nights = st.number_input("Minimum nights", 1, 365, 2)
    room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    property_type = st.text_input("Property type", value="Apartment")
    superhost = st.selectbox("Are you a Superhost?", ["No", "Yes"])

st.caption("📍 Location (defaults to central Paris)")
col3, col4 = st.columns(2)
with col3:
    latitude = st.number_input("Latitude", value=48.8566, format="%.4f")
with col4:
    longitude = st.number_input("Longitude", value=2.3522, format="%.4f")
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Advanced options (optional — we’ll fill sensible defaults)"):
    neighbourhood = st.text_input("Neighbourhood (e.g., Le Marais)", value="Le Marais")
    availability_365 = st.number_input("Availability (days per year)", 0, 365, 180)
    maximum_nights = st.number_input("Maximum nights allowed", 1, 1125, 365)
    min_nights_avg = st.number_input("Avg minimum nights (nearby)", 1, 365, min_nights)

# ---------- Predict ----------
if st.button("📅 Estimate Annual Availability", type="primary", use_container_width=True):
    row = {
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "beds": beds,
        "minimum_nights": min_nights,
        "maximum_nights": maximum_nights,
        "room_type": room_type,
        "property_type": property_type,
        "host_is_superhost": 1 if superhost == "Yes" else 0,
        "latitude": latitude,
        "longitude": longitude,
        "availability_365": availability_365,
        "minimum_nights_avg_ntm": min_nights_avg,
        "neighbourhood_cleansed": neighbourhood,
    }

    X = pd.DataFrame([row])
    # Enforce expected columns and fill missing if needed
    missing_cols = set(expected_cols) - set(X.columns)
    for col in missing_cols:
        if col in ["room_type", "property_type", "neighbourhood_cleansed"]:
            X[col] = "Unknown"
        elif col in ["latitude", "longitude"]:
            X[col] = [48.8566, 2.3522][0 if col == "latitude" else 1]
        elif col in ["host_is_superhost", "availability_365"]:
            X[col] = 0
        elif col in ["minimum_nights", "maximum_nights", "minimum_nights_avg_ntm"]:
            X[col] = 1
        else:
            X[col] = 0
    X = X[expected_cols]

    try:
        days_available = float(pipe.predict(X)[0])
        days_available = max(0.0, min(365.0, days_available))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✨ Estimated Availability (per year)")
        st.markdown(f"<div class='big-number'>{days_available:.0f} days</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(0.5, 365, height=0.25, color="#eee")
        ax.barh(0.5, days_available, height=0.25, color="#4CAF50")
        ax.set_xlim(0, 365)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Days available in a year")
        st.pyplot(fig, clear_figure=True)

        st.write("This is a directional estimate. Photos, reviews, and response time affect real demand.")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown('<div class="info-note">Tip: Shorter minimum stays and better photos often increase booking frequency.</div>', unsafe_allow_html=True)