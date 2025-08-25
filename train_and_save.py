# train_and_save.py
# -*- coding: utf-8 -*-

"""
Train & Save (Logistic Regression) for Streamlit
- Loads your dataset from Excel
- Builds preprocessing (scale numeric, one-hot categorical)
- Trains Logistic Regression
- Evaluates at threshold=0.40
- Saves a bundle to model.pkl (pipeline + metadata)
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


EXCEL_PATH = Path("Capstone Project - Raw Data (6).xlsx")  # must exist in the repo
TARGET_COL = "Success (0/1)"                                # adjust if your target name differs

# Columns used by the model (must exist in the Excel file)
NUM_COLS = ["Salary", "Time Working on Project", "Division Success Rate"]
CAT_COLS = ["Assignee", "Title", "Deadline", "Division", "Category", "Sub Category"]


def train_and_save():
    # ---- Load data ----
    df = pd.read_excel(EXCEL_PATH)

    # Keep only columns that actually exist (prevents key errors)
    num_cols = [c for c in NUM_COLS if c in df.columns and c != TARGET_COL]
    cat_cols = [c for c in CAT_COLS if c in df.columns and c != TARGET_COL]

    # ---- Split ----
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Preprocess & Model ----
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=5000))
    ])

    pipe.fit(X_train, y_train)

    # ---- Evaluate at threshold 0.40 ----
    threshold = 0.40
    proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_thr = (proba_test >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_thr)
    prec = precision_score(y_test, y_pred_thr, zero_division=0)
    rec = recall_score(y_test, y_pred_thr, zero_division=0)
    f1 = f1_score(y_test, y_pred_thr, zero_division=0)

    print("=== Evaluation @ threshold=0.40 ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_thr, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_thr))

    # ---- Save bundle ----
    cat_choices = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())[:100]
        for col in cat_cols
    }

    bundle = {
        "pipeline": pipe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_choices": cat_choices,
        "target_col": TARGET_COL,
        "threshold_default": threshold,
    }

    joblib.dump(bundle, "model.pkl", compress=3)
    print("✅ Saved model.pkl")


if __name__ == "__main__":
    train_and_save()
