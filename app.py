# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any

st.set_page_config(page_title="Salem Group - Proposal Success Predictive Model", page_icon="📈", layout="centered")

# ---- Top Title (white/grey aesthetic block) ----
st.markdown(
    """
    <div style="
        background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 22px;
        margin: 12px 0 18px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    ">
      <h1 style="margin:0; font-weight:700; color:#111827;">
        Salem Group - Proposal Success Predictive Model
      </h1>
      <div style="margin-top:8px; color:#4b5563; font-size:0.95rem; line-height:1.45;">
        Leveraging Machine Learning for an Automated Proposal Metrics Tool:
        Enhancing Proposal Success Tracking and Data-Driven Strategic Insights
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------- Load model bundle --------
@st.cache_resource
def load_bundle():
    bundle = joblib.load("model.pkl")
    pipe = bundle["pipeline"]
    num_cols: List[str] = bundle.get("num_cols", [])
    cat_cols: List[str] = bundle.get("cat_cols", [])
    cat_choices: Dict[str, List[str]] = bundle.get("cat_choices", {})
    threshold_default: float = float(bundle.get("threshold_default", 0.40))  # fixed 0.40 by default
    target_col: str = bundle.get("target_col", "Success (0/1)")

    # Optional lookup tables if your saved bundle includes them:
    assignee_avg_salary: Dict[str, float] = bundle.get("assignee_avg_salary", {})
    division_success_rate: Dict[str, float] = bundle.get("division_success_rate", {})

    return (
        pipe,
        num_cols,
        cat_cols,
        cat_choices,
        threshold_default,
        target_col,
        assignee_avg_salary,
        division_success_rate,
    )

(
    pipe,
    num_cols,
    cat_cols,
    cat_choices,
    threshold_default,
    target_col,
    assignee_avg_salary,
    division_success_rate,
) = load_bundle()

# =========================
# Proposal Success Estimator
# =========================
st.subheader("Proposal Success Estimator")

with st.form("single_row"):
    ui_vals: Dict[str, Any] = {}

    # Keep: Assignee, Division, Category, Sub Category, Time Working on Project
    left, right = st.columns(2)

    # Assignee
    with left:
        assignee_choices = cat_choices.get("Assignee", [])
        assignee = st.selectbox("Assignee", assignee_choices, index=0 if assignee_choices else None)
        ui_vals["Assignee"] = assignee

    # Division
    with right:
        division_choices = cat_choices.get("Division", [])
        division = st.selectbox("Division", division_choices, index=0 if division_choices else None)
        ui_vals["Division"] = division

    # Category / Sub Category (Title and Deadline removed)
    left2, right2 = st.columns(2)

    with left2:
        category_choices = cat_choices.get("Category", [])
        category = st.selectbox("Category", category_choices, index=0 if category_choices else None)
        ui_vals["Category"] = category

    with right2:
        subcat_choices = cat_choices.get("Sub Category", [])
        sub_category = st.selectbox("Sub Category", subcat_choices, index=0 if subcat_choices else None)
        ui_vals["Sub Category"] = sub_category

    # Time Working on Project FIRST (cannot be 0)
    time_work = st.number_input(
        "Time Working on Project",
        min_value=1.0,
        value=2.0,
        step=1.0,
        format="%.2f",
    )
    ui_vals["Time Working on Project"] = time_work

    # Auto-fill info
    st.caption("The fields below are auto-filled from your selections and are locked to prevent editing.")

    # Auto-fill: Salary from Assignee
    salary_autofill = float(assignee_avg_salary.get(assignee, 0.0)) if assignee else 0.0
    _ = st.number_input(
        "Salary (auto-filled from Assignee)",
        value=float(np.round(salary_autofill, 2)),
        step=0.0,
        disabled=True,
        help="Computed from the average salary of the selected assignee.",
    )
    ui_vals["Salary"] = salary_autofill

    # Auto-fill: Division Success Rate from Division
    div_sr_autofill = float(division_success_rate.get(division, 0.0)) if division else 0.0
    _ = st.number_input(
        "Division Success Rate (auto-filled from Division)",
        value=float(np.round(div_sr_autofill, 4)),
        step=0.0,
        disabled=True,
        help="Average success rate for the selected division.",
        format="%.4f",
    )
    ui_vals["Division Success Rate"] = div_sr_autofill

    submitted = st.form_submit_button("📈 Predict")

if submitted:
    X = pd.DataFrame([ui_vals])
    thr_fixed = float(threshold_default)  # use fixed backend threshold (0.40)

    if hasattr(pipe, "predict_proba"):
        prob = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(prob >= thr_fixed)
        st.metric("Probability (class = 1)", f"{prob:.4f}")
        st.write(f"**Prediction:** {pred} (threshold fixed at {thr_fixed:.2f})")
    else:
        yhat = pipe.predict(X)[0]
        st.write(f"**Prediction:** {yhat}")

# =============
# Batch Scoring
# =============
st.divider()
st.subheader("Batch Scoring")

uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    with st.spinner("Scoring..."):
        if hasattr(pipe, "predict_proba"):
            df["proba"] = pipe.predict_proba(df)[:, 1]
            thr_fixed = float(threshold_default)  # same fixed threshold 0.40
            df["prediction"] = (df["proba"] >= thr_fixed).astype(int)
        else:
            df["prediction"] = pipe.predict(df)

    st.dataframe(df.head(50))
    st.download_button("Download predictions.csv", df.to_csv(index=False).encode(), "predictions.csv")

# Optional: quick debug panel
with st.expander("Debug: model & column info"):
    st.write({
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_col": target_col,
        "threshold_fixed": threshold_default,
    })
