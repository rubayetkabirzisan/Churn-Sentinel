# src/baseline_model.py
# Churn Sentinel — Baseline: Logistic Regression
# Run: python src/baseline_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"
FIGURES_DIR   = "reports/figures"
REPORT_PATH   = "reports/model_eval.md"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────
COLORS = {
    "churn"    : "#E63946",
    "no_churn" : "#2A9D8F",
    "accent"   : "#E9C46A",
    "bg"       : "#1A1A2E",
    "text"     : "#EAEAEA"
}
plt.rcParams.update({
    "figure.facecolor" : COLORS["bg"],
    "axes.facecolor"   : COLORS["bg"],
    "axes.edgecolor"   : COLORS["text"],
    "axes.labelcolor"  : COLORS["text"],
    "xtick.color"      : COLORS["text"],
    "ytick.color"      : COLORS["text"],
    "text.color"       : COLORS["text"],
    "font.family"      : "monospace",
    "grid.color"       : "#2E2E4E",
    "grid.alpha"       : 0.5
})


# ════════════════════════════════════════════════════════
# LOAD PROCESSED DATA
# ════════════════════════════════════════════════════════
def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    print(f"✅ Data loaded — train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════
# TRAIN BASELINE
# ════════════════════════════════════════════════════════
def train_baseline(X_train, y_train):
    # Handle class imbalance with class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    model = LogisticRegression(
        max_iter     = 1000,
        class_weight = class_weight_dict,
        random_state = 42,
        solver       = "lbfgs"
    )
    model.fit(X_train, y_train)
    print("✅ Logistic Regression trained")
    return model


# ════════════════════════════════════════════════════════
# EVALUATE
# ════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test, model_name="Logistic Regression"):
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model"     : model_name,
        "accuracy"  : round(accuracy_score(y_test,  y_pred),  4),
        "precision" : round(precision_score(y_test, y_pred),  4),
        "recall"    : round(recall_score(y_test,    y_pred),  4),
        "f1"        : round(f1_score(y_test,        y_pred),  4),
        "roc_auc"   : round(roc_auc_score(y_test,  y_prob),  4),
    }

    print(f"\n{'='*55}")
    print(f"   {model_name} — Evaluation Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']  * 100:.2f}%")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall']    * 100:.2f}%")
    print(f"  F1-Score  : {metrics['f1']        * 100:.2f}%")
    print(f"  ROC-AUC   : {metrics['roc_auc']   * 100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["No Churn", "Churn"]))

    return metrics, y_pred, y_prob


# ════════════════════════════════════════════════════════
# PLOT 1 — Confusion Matrix
# ════════════════════════════════════════════════════════
def plot_confusion_matrix(y_test, y_pred, model_name, filename):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    im = ax.imshow(cm, interpolation="nearest",
                   cmap=plt.cm.RdYlGn)
    plt.colorbar(im, ax=ax)

    classes = ["No Churn", "Churn"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold",
                    color="white" if cm[i,j] < thresh else "black")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Confusion matrix saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# PLOT 2 — ROC Curve
# ════════════════════════════════════════════════════════
def plot_roc_curve(model, X_test, y_test, model_name, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"ROC Curve — {model_name}",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax,
        color=COLORS["churn"], lw=2,
        name=model_name
    )
    ax.plot([0,1],[0,1], "--", color=COLORS["accent"],
            linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ ROC curve saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# PLOT 3 — Top Feature Coefficients
# ════════════════════════════════════════════════════════
def plot_feature_coefficients(model, feature_names, filename):
    coefs = pd.Series(model.coef_[0], index=feature_names)
    top   = coefs.abs().nlargest(15).index
    top_coefs = coefs[top].sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Top 15 Feature Coefficients — Logistic Regression",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    colors_bar = [COLORS["churn"] if v > 0
                  else COLORS["no_churn"] for v in top_coefs.values]
    ax.barh(top_coefs.index, top_coefs.values,
            color=colors_bar, edgecolor="#FFFFFF", linewidth=0.3)
    ax.axvline(x=0, color=COLORS["text"], linewidth=1)
    ax.set_xlabel("Coefficient Value")

    churn_patch    = mpatches.Patch(color=COLORS["churn"],
                                    label="→ increases churn risk")
    no_churn_patch = mpatches.Patch(color=COLORS["no_churn"],
                                    label="→ decreases churn risk")
    ax.legend(handles=[churn_patch, no_churn_patch])

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Coefficients plot saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# SAVE METRICS TO REPORT
# ════════════════════════════════════════════════════════
def save_metrics_to_report(metrics: dict):
    os.makedirs("reports", exist_ok=True)

    # Load existing report or start fresh
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            existing = f.read()
    else:
        existing = "# Churn Sentinel — Model Evaluation Report\n\n"

    # Append baseline section
    section = f"""
## Baseline: {metrics['model']}

| Metric    | Score  |
|-----------|--------|
| Accuracy  | {metrics['accuracy']*100:.2f}% |
| Precision | {metrics['precision']*100:.2f}% |
| Recall    | {metrics['recall']*100:.2f}% |
| F1-Score  | {metrics['f1']*100:.2f}% |
| ROC-AUC   | {metrics['roc_auc']*100:.2f}% |

**Figures:**
- `reports/figures/06_cm_baseline.png`
- `reports/figures/07_roc_baseline.png`
- `reports/figures/08_coef_baseline.png`

---
"""
    with open(REPORT_PATH, "w") as f:
        f.write(existing + section)

    # Also save as JSON for dashboard use
    joblib.dump(metrics, f"{OUTPUTS_DIR}/baseline_metrics.pkl")
    print(f"✅ Metrics saved → {REPORT_PATH}")
    print(f"✅ Metrics pkl   → {OUTPUTS_DIR}/baseline_metrics.pkl")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def run_baseline():
    print("\n" + "="*55)
    print("   CHURN SENTINEL — Baseline Model Training")
    print("="*55 + "\n")

    # Load
    X_train, X_test, y_train, y_test = load_data()
    feature_names = pd.read_csv(
        f"{PROCESSED_DIR}/X_train.csv").columns.tolist()

    # Train
    model = train_baseline(X_train, y_train)

    # Evaluate
    metrics, y_pred, y_prob = evaluate(
        model, X_test, y_test, "Logistic Regression")

    # Plots
    plot_confusion_matrix(
        y_test, y_pred,
        "Logistic Regression", "06_cm_baseline.png")

    plot_roc_curve(
        model, X_test, y_test,
        "Logistic Regression", "07_roc_baseline.png")

    plot_feature_coefficients(
        model, feature_names, "08_coef_baseline.png")

    # Save
    joblib.dump(model, f"{OUTPUTS_DIR}/baseline_model.pkl")
    save_metrics_to_report(metrics)

    print("\n" + "="*55)
    print("✅ BASELINE COMPLETE — model saved to outputs/")
    print("="*55)

    return model, metrics


if __name__ == "__main__":
    run_baseline()