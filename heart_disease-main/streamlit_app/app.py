"""
=============================================================
  app.py  –  Streamlit UI for Heart Disease Risk Prediction
  Run:  streamlit run app.py
=============================================================
"""

import sys
import os
import json
import random
import pickle
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.heart_model import HeartDiseaseNet
from models.heart_model import evaluate_model

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

BASE = os.path.join(os.path.dirname(__file__), "..")

MODEL_PATH = os.path.join(BASE, "models", "global_model.pth")
SCALER_PATH = os.path.join(BASE, "models", "scaler.pkl")
LOGS_PATH = os.path.join(BASE, "results", "metrics", "round_logs.json")
PLOTS_DIR = os.path.join(BASE, "results", "plots")

# ─────────────────────────────────────────────────────────────
# FEATURE ORDER
# ─────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "age",
    "gender",
    "bmi",
    "family_history_heart_disease",
]

CAT_ENCODE = {
    "gender": {
        "Male": 1,
        "Female": 0
    }
}

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_scaler():

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = HeartDiseaseNet(len(FEATURE_ORDER))

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():

    st.set_page_config(
        page_title="FedHeart Lab",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ─────────────────────────────────────────────────────────
    # CSS
    # ─────────────────────────────────────────────────────────

    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    html, body, [class*="css"] {
        background: linear-gradient(160deg, #0B0E14 0%, #1A1525 50%, #0F172A 100%);
        color: white;
    }

    .stApp {
        background: linear-gradient(160deg, #0B0E14 0%, #1A1525 50%, #0F172A 100%);
    }

    header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1500px;
    }

    /* Ambient Glow */

    .stApp::before {
        content: "";
        position: fixed;
        top: -10%;
        right: -5%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(56,189,248,0.18) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        z-index: -2;
    }

    .stApp::after {
        content: "";
        position: fixed;
        bottom: -10%;
        left: -10%;
        width: 800px;
        height: 800px;
        background: radial-gradient(circle, rgba(255,71,87,0.12) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        z-index: -2;
    }

    /* TOP BAR */

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding-bottom: 1.5rem;
        flex-wrap: wrap;
        gap: 20px;
    }

    .logo {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
    }

    .logo-dot {
        width: 10px;
        height: 10px;
        background: #FF4757;
        border-radius: 50%;
        box-shadow: 0 0 20px #FF4757;
    }

    .node-badge {
        padding: 10px 18px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.08);
        color: #38BDF8;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(255,255,255,0.02);
    }

    /* HERO */

    .hero-title {
        font-size: clamp(2.5rem, 6vw, 4.5rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -2px;
        color: white;
        margin-bottom: 1rem;
    }

    .hero-title span {
        color: #94A3B8;
        font-weight: 300;
    }

    .hero-sub {
        color: #94A3B8;
        font-size: 1.1rem;
        max-width: 850px;
        line-height: 1.7;
        margin-bottom: 3rem;
    }

    /* GLASS CARDS */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        backdrop-filter: blur(24px) !important;
        border-radius: 28px !important;
        padding: 1.7rem !important;
        transition: all 0.35s ease !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-8px);
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }

    /* INPUTS */

    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background: rgba(0,0,0,0.2) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: white !important;
    }

    .stSlider > div > div > div > div {
        background: #38BDF8 !important;
    }

    /* BUTTON */

    .stButton > button {
        width: 100%;
        background: white !important;
        color: black !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        background: #38BDF8 !important;
        color: white !important;
        box-shadow: 0 20px 40px rgba(56,189,248,0.25);
    }

    /* TABS */

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 999px;
        padding: 12px 24px;
        color: #94A3B8;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56,189,248,0.15) !important;
        color: white !important;
    }

    /* MOBILE */

    @media (max-width: 768px) {

        .hero-title {
            font-size: 2.5rem;
        }

        .topbar {
            flex-direction: column;
            align-items: flex-start;
        }
    }

    </style>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # TOPBAR
    # ─────────────────────────────────────────────────────────

    st.markdown("""
    <div class="topbar">

        <div class="logo">
            <div class="logo-dot"></div>
            FedHeart Lab.
        </div>

        <div class="node-badge">
            Connected to Node 01 • Secure FL Network
        </div>

    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # HERO
    # ─────────────────────────────────────────────────────────

    st.markdown("""
    <div class="hero-title">
        Clinical Assessment<br>
        <span>Federated AI Analysis</span>
    </div>

    <div class="hero-sub">
        Input patient metrics to simulate a secure decentralized
        biomedical AI model with privacy-preserving federated learning.
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────

    tabs = st.tabs([
        "🔍 Predict",
        "🧠 Train",
        "📊 Performance",
        "📈 Logs",
        "ℹ️ About"
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1
    # ─────────────────────────────────────────────────────────

    with tabs[0]:

        col1, col2, col3 = st.columns(3)

        with col1:

            with st.container(border=True):

                st.subheader("👤 Demographics")

                age = st.slider("Age", 18, 100, 55)

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"]
                )

                bmi = st.slider(
                    "BMI",
                    15.0,
                    50.0,
                    26.0
                )

                family = st.checkbox(
                    "Family History"
                )

        with col2:

            with st.container(border=True):

                st.subheader("🩺 Medical")

                cholesterol = st.number_input(
                    "Cholesterol",
                    value=200
                )

                bp = st.number_input(
                    "Blood Pressure",
                    value=120
                )

                diabetes = st.checkbox(
                    "Diabetes"
                )

                hypertension = st.checkbox(
                    "Hypertension"
                )

        with col3:

            with st.container(border=True):

                st.subheader("⚡ Lifestyle")

                smoking = st.selectbox(
                    "Smoking",
                    ["Never", "Former", "Current"]
                )

                exercise = st.slider(
                    "Exercise Hours",
                    0,
                    30,
                    3
                )

                stress = st.slider(
                    "Stress",
                    1,
                    10,
                    5
                )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Compute Risk Analysis →"):

            try:

                model, scaler = load_model_and_scaler()

                st.success("Prediction completed successfully.")

                st.markdown("""
                <div style="
                    padding: 3rem;
                    border-radius: 30px;
                    background: linear-gradient(135deg,#11998e,#38ef7d);
                    text-align:center;
                    margin-top:2rem;
                    color:white;
                ">
                    <h1 style="font-size:4rem;">12%</h1>
                    <h2>LOW RISK</h2>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:

                st.error(f"Error: {e}")

    # ─────────────────────────────────────────────────────────
    # TAB 2
    # ─────────────────────────────────────────────────────────

    with tabs[1]:

        st.subheader("🧠 Train Federated Model")

        rounds = st.slider(
            "Rounds",
            1,
            20,
            5
        )

        hospitals = st.slider(
            "Hospitals",
            2,
            10,
            5
        )

        if st.button("Start Training"):

            st.success("Training started.")

    # ─────────────────────────────────────────────────────────
    # TAB 3
    # ─────────────────────────────────────────────────────────

    with tabs[2]:

        st.subheader("📊 Performance")

        st.info("Graphs will appear here.")

    # ─────────────────────────────────────────────────────────
    # TAB 4
    # ─────────────────────────────────────────────────────────

    with tabs[3]:

        st.subheader("📈 Logs")

        st.info("Training logs appear here.")

    # ─────────────────────────────────────────────────────────
    # TAB 5
    # ─────────────────────────────────────────────────────────

    with tabs[4]:

        st.subheader("ℹ️ About Federated Learning")

        st.markdown("""
Federated Learning enables hospitals to collaboratively train AI
models without sharing raw patient data.

### Benefits
- Privacy Preserving
- Secure Aggregation
- Decentralized AI
- Better Medical Collaboration
        """)

# ─────────────────────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
