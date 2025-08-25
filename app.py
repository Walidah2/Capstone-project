# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any

st.set_page_config(page_title="Capstone Predictor", page_icon="📊", layout="centered")

# -------- Helper: align input data to training schema --------
def align_columns(df: pd.DataFrame, num_cols, cat_cols, target_col):
    """Make incoming data match the training schema used in training."""
    df = df.copy()

    # 1) Drop target if included
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # 2) Ensure numeric columns exist and are numeric
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Ensure categorical columns exist and are strings
    for c in cat_cols:
        if c not in df.columns:
            df[c] = ""
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
        else:
            df[c] = df[c].astype(str)

    # 4) Fill missing numeric values
    df[num_cols] = df[num_cols].fillna(0.0)

    return df

# -------- Load model bundle --------
@st.cache_resource
def load_bundle():
    bundle = joblib.load("model.pkl")
    pipe = bundle["pipeline"]
    num_cols: List[str] = bundle.get("num_cols", [])
    cat_cols: List[str] = bundle.get("cat_cols", [])
    cat_choices: Dict[str, List[str]] = bundle.get("cat_choices", {})
    threshold_default: float = float(bundle.get("threshold_default", 0.40))
    target_col: str = bundle.get("target_col", "Success (0/1)")
    return pipe, num_cols, cat_cols, cat_choices, threshold_default, target_col

pipe, num_cols, cat_cols, cat_choices, threshold_default, target_col = load_bundle()

st.title("📊 Capstone Predictive Model")
st.caption("Upload CSV for batch scoring or enter a single row below.")

# -------- Batch predictions --------
uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
    X = align_columns(raw, num_cols, cat_cols, target_col)

    with st.spinner("Scoring..."):
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)[:, 1]
            thr_batch = st.slider("Decision threshold (batch)", 0.0, 1.0, float(threshold_default), 0.01, key="thr_batch")
            pred = (proba >= thr_batch).astype(int)
            out = raw.copy()
            out["proba"] = proba
            out["prediction"] = pred
        else:
            pred = pipe.predict(X)
            out = raw.copy()
            out["prediction"] = pred

    st.dataframe(out.head(50))
    st.download_button("Download predictions.csv", out.to_csv(index=False).encode(), "predictions.csv")

    # Offer a CSV template with the correct schema
    template = pd.DataFrame(columns=num_cols + cat_cols)
    st.download_button("Download template.csv", template.to_csv(index=False).encode(), "template.csv")

st.divider()
st.subheader("Single-row prediction")

# -------- Single-row form --------
with st.form("single_row"):
    ui_vals: Dict[str, Any] = {}

    if num_cols:
        st.markdown("**Numeric features**")
        for col in num_cols:
            ui_vals[col] = st.number_input(col, value=0.0, step=1.0, format="%.4f")

    if cat_cols:
        st.markdown("**Categorical features**")
        for col in cat_cols:
            choices = cat_choices.get(col)
            if choices and len(choices) > 0:
                ui_vals[col] = st.selectbox(col, choices, index=0)
            else:
                ui_vals[col] = st.text_input(col, "")

    thr_single = st.slider("Decision threshold (single)", 0.0, 1.0, float(threshold_default), 0.01, key="thr_single")
    submitted = st.form_submit_button("Predict")

if submitted:
    X = align_columns(pd.DataFrame([ui_vals]), num_cols, cat_cols, target_col)

    if hasattr(pipe, "predict_proba"):
        prob = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(prob >= thr_single)
        st.metric("Probability (class = 1)", f"{prob:.4f}")
        st.write(f"**Prediction:** {pred} (threshold = {thr_single:.2f})")
    else:
        yhat = pipe.predict(X)[0]
        st.write(f"**Prediction:** {yhat}")

with st.expander("Debug: model & column info"):
    st.write({"num_cols": num_cols, "cat_cols": cat_cols, "target_col": target_col})
