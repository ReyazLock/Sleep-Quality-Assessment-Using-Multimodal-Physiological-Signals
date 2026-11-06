#!/usr/bin/env python3
"""
Demo: predict sleep-disordered breathing (SDB) from physiological signals
with classic machine learning (scikit-learn).

Install (inside your venv):
    pip install pandas matplotlib          # scikit-learn is already installed

Run:
    python sleep_ml_demo.py enhanced_sleep_disordered_breathing_dataset.csv

What it does
    1. Loads the CSV and prints a short data summary.
    2. Uses the signals + demographics as features and "Diagnosis of SDB"
       (Yes/No) as the target.
    3. Compares a few models with Leave-One-Out cross-validation
       (the dataset is tiny, so a normal train/test split would be unreliable).
    4. Prints metrics, a confusion matrix, feature importances and the
       per-patient predictions; saves a CSV and a PNG chart.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Diagnosis of SDB"
ID_COL = "Patient ID"
# Not used as features: they come from the diagnosis itself (leakage) or are free text.
EXCLUDE = [TARGET, "Severity", "Treatment", "Physician Notes",
           "Patient Symptoms", ID_COL]
AHI_COL = "Apnea-Hypopnea Index (AHI)"
SEED = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"[error] File not found: {path}")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        sys.exit(f"[error] Column '{TARGET}' not found in the CSV.")
    return df


def print_summary(df: pd.DataFrame, features: list) -> None:
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Patients: {len(df)} | Features used: {len(features)}")
    print(f"Target '{TARGET}': {df[TARGET].value_counts().to_dict()}")
    print("\nMean of numeric features by diagnosis:")
    num = df[features].select_dtypes("number").columns
    print(df.groupby(TARGET)[list(num)].mean().round(2).T.to_string())
    if len(df) < 30:
        print("\n[note] Very few rows: treat every result below as a demo only.")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


def make_models(X: pd.DataFrame) -> dict:
    def pipe(clf):
        return Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])
    return {
        "Majority-class baseline": pipe(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": pipe(LogisticRegression(max_iter=1000,
                                                       class_weight="balanced")),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=200,
                                                     class_weight="balanced",
                                                     random_state=SEED)),
    }


def score(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        out["ROC AUC"] = roc_auc_score(y_true, y_prob)
    return out


def feature_names(fitted_pipe: Pipeline) -> list:
    names = fitted_pipe.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[-1] for n in names]


def make_plot(results: pd.DataFrame, importances: pd.Series, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # save to file, no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed -> skipping the chart "
              "(pip install matplotlib).")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    results["Accuracy"].plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Leave-One-Out accuracy by model")
    axes[0].set_xlabel("Accuracy")
    importances.sort_values().tail(10).plot.barh(ax=axes[1], color="darkorange")
    axes[1].set_title("Top features (Random Forest importance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scikit-learn SDB demo")
    ap.add_argument("csv", type=Path, help="Input dataset (.csv)")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="Folder for the output CSV/PNG (default: current folder)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    features = [c for c in df.columns if c not in EXCLUDE]
    X = df[features]
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)

    print_summary(df, features)

    # ---- Model comparison with Leave-One-Out CV -------------------------
    cv = LeaveOneOut()
    models = make_models(X)
    rows, preds, probs = {}, {}, {}
    for name, model in models.items():
        preds[name] = cross_val_predict(model, X, y, cv=cv)
        probs[name] = cross_val_predict(model, X, y, cv=cv,
                                        method="predict_proba")[:, 1]
        rows[name] = score(y, preds[name], probs[name])

    # Simple clinical rule for reference: AHI >= 5 means SDB
    if AHI_COL in df.columns:
        rule_pred = (df[AHI_COL] >= 5).astype(int)
        rows["Clinical rule (AHI >= 5)"] = score(y, rule_pred, df[AHI_COL])
        preds["Clinical rule (AHI >= 5)"] = rule_pred

    results = pd.DataFrame(rows).T
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Leave-One-Out cross-validation)")
    print("=" * 60)
    print(results.round(3).to_string())

    ml_only = results.drop(index=["Majority-class baseline",
                                  "Clinical rule (AHI >= 5)"], errors="ignore")
    best = ml_only["F1"].idxmax()
    print(f"\nBest ML model by F1: {best}")
    cm = confusion_matrix(y, preds[best], labels=[0, 1])
    print("Confusion matrix (rows = real, cols = predicted):")
    print(pd.DataFrame(cm, index=["real No", "real Yes"],
                       columns=["pred No", "pred Yes"]).to_string())

    # ---- Feature importance (fit on all data, for interpretation) -------
    rf = models["Random Forest"].fit(X, y)
    importances = pd.Series(rf.named_steps["clf"].feature_importances_,
                            index=feature_names(rf)).sort_values(ascending=False)
    print("\nTop features (Random Forest importance):")
    print(importances.head(8).round(3).to_string())

    lr = models["Logistic Regression"].fit(X, y)
    coefs = pd.Series(lr.named_steps["clf"].coef_[0],
                      index=feature_names(lr)).sort_values(key=abs, ascending=False)
    print("\nTop features (Logistic Regression coefficient; + pushes toward SDB):")
    print(coefs.head(8).round(3).to_string())

    # ---- Per-patient feedback ------------------------------------------
    report = pd.DataFrame({
        ID_COL: df.get(ID_COL, df.index + 1),
        "Real diagnosis": df[TARGET],
        "Predicted": np.where(preds[best] == 1, "Yes", "No"),
        "P(SDB)": probs[best].round(3),
    })
    report["Correct"] = np.where((preds[best] == y).values, "OK", "WRONG")
    print("\n" + "=" * 60)
    print(f"PER-PATIENT PREDICTIONS ({best}, each patient predicted while held out)")
    print("=" * 60)
    print(report.to_string(index=False))

    csv_path = args.out_dir / "ml_predictions.csv"
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nPredictions saved to {csv_path}")
    make_plot(results, importances, args.out_dir / "ml_summary.png")


if __name__ == "__main__":
    main()
