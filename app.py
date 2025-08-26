# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any

# ----------------------- Page Setup & Style -----------------------
st.set_page_config(
    page_title="Salem Group - Proposal Success Predictive Model",
    page_icon="📈",
    layout="centered"
)

# Minimal, safe CSS polish
st.markdown(
    """
    <style>
      .app-header {
        padding: 1.2rem 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
        color: white;
        text-align: center;
        margin-bottom: 1rem;
      }
      .app-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 1rem 1rem 0.5rem 1rem;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
      }
      .caption-muted {
        color: #6b7280;
        font-size: 0.92rem;
      }
      .af-badge {
        display: inline-block;
        font-size: 0.78rem;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #e0e7ff;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        margin-left: 0.4rem;
        vertical-align: middle;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-header">
      <h2 style="margin:0;">Salem Group – Proposal Success Predictive Model</h2>
      <div style="opacity:0.95; margin-top:6px;">Batch scoring & single-row predictions with smart auto-fill</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------- Load Model Bundle -----------------------
@st.cache_resource
def load_bundle():
    bundle = joblib.load("model.pkl")
    pipe = bundle["pipeline"]
    num_cols: List[str] = bundle.get("num_cols", [])
    cat_cols: List[str] = bundle.get("cat_cols", [])
    cat_choices: Dict[str, List[str]] = bundle.get("cat_choices", {})
    avg_salary_by_assignee: Dict[str, float] = bundle.get("avg_salary_by_assignee", {})
    div_success_by_division: Dict[str, float] = bundle.get("div_success_by_division", {})
    threshold_default: float = float(bundle.get("threshold_default", 0.40))
    target_col: str = bundle.get("target_col", "Success (0/1)")
    return (pipe, num_cols, cat_cols, cat_choices,
            avg_salary_by_assignee, div_success_by_division,
            threshold_default, target_col)

(pipe,
 num_cols, cat_cols, cat_choices,
 avg_salary_by_assignee, div_success_by_division,
 threshold_default, target_col) = load_bundle()

# Model expectations
expected_num = set(num_cols)
expected_cat = set(cat_cols)

# Hide these from UI (but still pass safe placeholders to the model)
HIDE_CATS = {"Deadline", "Title"}
PLACEHOLDER = "(missing)"

# ----------------------- Helpers -----------------------
def ensure_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all model-required columns exist and apply auto-fill rules."""
    df = df.copy()

    # Ensure required columns exist
    for c in expected_cat:
        if c not in df.columns:
            df[c] = PLACEHOLDER
    for c in expected_num:
        if c not in df.columns:
            df[c] = 0.0

    # Force hidden cats to placeholder
    for c in HIDE_CATS:
        if c in df.columns:
            df[c] = df[c].fillna(PLACEHOLDER).replace("", PLACEHOLDER)

    # Auto-fill Division Success Rate from Division when empty/0
    if "Division" in df.columns and "Division Success Rate" in df.columns:
        def _fill_div(row):
            v = row.get("Division Success Rate", np.nan)
            if pd.isna(v) or (isinstance(v, (int, float)) and float(v) == 0.0):
                return div_success_by_division.get(str(row.get("Division")), v)
            return v
        df["Division Success Rate"] = df.apply(_fill_div, axis=1)

    # Auto-fill Salary from Assignee when empty/0
    if "Assignee" in df.columns and "Salary" in df.columns:
        def _fill_sal(row):
            v = row.get("Salary", np.nan)
            if pd.isna(v) or (isinstance(v, (int, float)) and float(v) == 0.0):
                return avg_salary_by_assignee.get(str(row.get("Assignee")), v)
            return v
        df["Salary"] = df.apply(_fill_sal, axis=1)

    return df

# ----------------------- Batch Scoring -----------------------
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("📦 Batch Scoring (CSV)")
    st.markdown(
        "<div class='caption-muted'>Upload a CSV. Missing columns will be added. "
        "Salary and Division Success Rate will be auto-filled from Assignee/Division.</div>",
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        df_scored = ensure_model_columns(df_in)

        with st.spinner("Scoring..."):
            if hasattr(pipe, "predict_proba"):
                df_scored["proba"] = pipe.predict_proba(df_scored)[:, 1]
                thr_batch = st.slider("Decision threshold (batch)", 0.0, 1.0,
                                      float(threshold_default), 0.01, key="thr_batch")
                df_scored["prediction"] = (df_scored["proba"] >= thr_batch).astype(int)
            else:
                df_scored["prediction"] = pipe.predict(df_scored)

        st.markdown("**Preview**")
        st.dataframe(df_scored.head(50), use_container_width=True)
        st.download_button("Download predictions.csv",
                           df_scored.to_csv(index=False).encode(),
                           "predictions.csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Single Row -----------------------
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("🧮 Single-Row Prediction")

    # Prepare choices
    assignee_choices = cat_choices.get("Assignee", [])
    division_choices = cat_choices.get("Division", [])
    category_choices = cat_choices.get("Category", [])
    subcat_choices = cat_choices.get("Sub Category", [])

    with st.form("single_row"):
        ui: Dict[str, Any] = {}

        # Visible categoricals (we hide Title/Deadline)
        c1, c2 = st.columns(2)
        with c1:
            ui["Assignee"] = st.selectbox(
                "Assignee",
                assignee_choices,
                index=0 if assignee_choices else None
            )
        with c2:
            ui["Division"] = st.selectbox(
                "Division",
                division_choices,
                index=0 if division_choices else None
            )

        c3, c4 = st.columns(2)
        with c3:
            ui["Category"] = st.selectbox(
                "Category",
                category_choices,
                index=0 if category_choices else None
            )
        with c4:
            ui["Sub Category"] = st.selectbox(
                "Sub Category",
                subcat_choices,
                index=0 if subcat_choices else None
            )

        st.markdown(
            "<div class='caption-muted'>"
            "The fields below are <b>auto-filled</b> from your selections and are locked to prevent editing."
            "</div>",
            unsafe_allow_html=True
        )

        # Manual numeric
        ui["Time Working on Project"] = st.number_input(
            "Time Working on Project",
            value=0.0,
            step=1.0
        )

        # Auto-filled (locked): Salary from Assignee
        assignee = ui.get("Assignee")
        salary_default = float(avg_salary_by_assignee.get(str(assignee), 0.0)) if assignee is not None else 0.0
        st.number_input(
            "Salary (auto-filled from Assignee)  ",
            value=salary_default, step=100.0, disabled=True, key="salary_display"
        )
        ui["Salary"] = salary_default  # still pass value to the model

        # Auto-filled (locked): DSR from Division
        division = ui.get("Division")
        dsr_default = float(div_success_by_division.get(str(division), 0.0)) if division is not None else 0.0
        st.number_input(
            "Division Success Rate (auto-filled from Division) ",
            value=dsr_default, step=0.01, disabled=True, format="%.4f", key="dsr_display"
        )
        ui["Division Success Rate"] = dsr_default  # pass to the model

        # Hidden cats → placeholders
        for hc in HIDE_CATS:
            if hc in expected_cat:
                ui[hc] = PLACEHOLDER

        thr_single = st.slider("Decision threshold (single)", 0.0, 1.0,
                               float(threshold_default), 0.01, key="thr_single")
        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        X = pd.DataFrame([ui])
        X = ensure_model_columns(X)

        if hasattr(pipe, "predict_proba"):
            p = float(pipe.predict_proba(X)[:, 1][0])
            pred = int(p >= thr_single)
            st.metric("Probability (class = 1)", f"{p:.4f}")
            st.success(f"**Prediction:** {pred}  —  threshold = {thr_single:.2f}")
        else:
            yhat = pipe.predict(X)[0]
            st.success(f"**Prediction:** {yhat}")

    with st.expander("Debug: model & column info"):
        st.write({
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "hidden_cat_placeholders": list(HIDE_CATS),
            "has_avg_salary_by_assignee": len(avg_salary_by_assignee) > 0,
            "has_div_success_by_division": len(div_success_by_division) > 0,
            "target_col": target_col
        })

    st.markdown('</div>', unsafe_allow_html=True)

st.caption("© Salem Group")
