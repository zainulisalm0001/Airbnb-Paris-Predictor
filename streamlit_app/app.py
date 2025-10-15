# streamlit_app/app.py

import base64
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Paris Airbnb — Smart Pricing", page_icon="🗼", layout="centered")

# ---------- Custom Styles ----------
def inject_css():
    st.markdown("""
        <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(180deg, #fafafa 0%, #f7fbff 35%, #fff 100%);
            color: #222 !important;
        }
        /* Hero title */
        .hero-title {
            font-size: 36px;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            font-size: 16px;
            color: #666;
            margin-bottom: 1.25rem;
        }
        .card {
            border-radius: 14px;
            padding: 20px;
            background: #ffffff;
            border: 1px solid #eaeaef;
            box-shadow: 0px 4px 14px rgba(0,0,0,0.04);
            margin: 0.75rem 0;
        }
        .pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: #f0f6ff;
            border: 1px solid #e0ecff;
            color: #2255aa;
            font-weight: 600;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .footer-note {
            color: #666;
            font-size: 13px;
        }
        </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------- Header ----------
st.markdown('<div class="pill">PARIS AIRBNB</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Smart Pricing & Availability</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Estimate a listing’s nightly price and get a hint of how often it may be available — '
    'powered by data from Paris Airbnb listings.</div>',
    unsafe_allow_html=True
)

# ---------- Cards ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 💶 Price Predictor")
st.write(
    "Not sure how to price your place? Enter a few details (like bedrooms, room type, and location), "
    "and we’ll estimate a fair nightly price based on similar places in Paris."
)
st.page_link("pages/1_Price_Predictor.py", label="Open Price Predictor →", icon="💶")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📅 Availability Predictor")
st.write(
    "Curious about demand? This tool predicts annual availability trends — how many days your listing "
    "might be open for booking — based on its attributes."
)
st.page_link("pages/2_Availability_Predictor.py", label="Open Availability Predictor →", icon="📅")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    '<div class="footer-note">Note: These are estimates based on historical data. Actual results vary with season, events, photos, reviews, and host responsiveness.</div>',
    unsafe_allow_html=True
)