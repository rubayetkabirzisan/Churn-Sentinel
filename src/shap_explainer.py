# src/shap_explainer.py
# Churn Sentinel — SHAP Explainability Layer
# Run: python src/shap_explainer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import shap

# ── Paths ─────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"
FIGURES_DIR   = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────
COLORS = {
    "bg"   : "#1A1A2E",
    "text" : "#EAEAEA",
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
})


# ════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ════════════════════════════════════════════════════════
def load_artifacts():
    model         = joblib.load(f"{OUTPUTS_DIR}/xgb_model.pkl")
    feature_names = joblib.load(f"{OUTPUTS_DIR}/feature_names.pkl")
    X_test        = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test        = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()

    print(f"✅ Model loaded    : XGBoost")
    print(f"✅ Test set loaded : {X_test.shape[0]:,} samples × "
          f"{X_test.shape[1]} features")
    return model, X_test, y_test, feature_names


# ════════════════════════════════════════════════════════
# BUILD SHAP EXPLAINER + VALUES
# ════════════════════════════════════════════════════════
def build_shap_explainer(model, X_test):
    print("\n🔧 Building SHAP TreeExplainer...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)          # Explanation object
    print(f"✅ SHAP values computed — shape: {shap_values.values.shape}")
    return explainer, shap_values


# ════════════════════════════════════════════════════════
# PLOT 1 — Global Summary Plot (Beeswarm)
# ════════════════════════════════════════════════════════
def plot_summary_beeswarm(shap_values, X_test, filename):
    print("\n📊 Generating SHAP Summary Plot (Beeswarm)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS["bg"])

    shap.summary_plot(
        shap_values.values,
        X_test,
        plot_type = "dot",
        max_display = 20,
        show = False,
        color_bar = True
    )

    plt.title("SHAP Summary — Global Feature Impact on Churn",
              fontsize=13, fontweight="bold",
              color=COLORS["text"], pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}",
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.show()
    print(f"✅ Beeswarm plot saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# PLOT 2 — Global Bar Plot (Mean |SHAP|)
# ════════════════════════════════════════════════════════
def plot_summary_bar(shap_values, X_test, filename):
    print("\n📊 Generating SHAP Bar Plot (Mean |SHAP|)...")
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(COLORS["bg"])

    shap.summary_plot(
        shap_values.values,
        X_test,
        plot_type   = "bar",
        max_display = 15,
        show        = False,
        color       = "#F4A261"
    )

    plt.title("SHAP Feature Importance — Mean |SHAP Value|",
              fontsize=13, fontweight="bold",
              color=COLORS["text"], pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}",
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.show()
    print(f"✅ Bar plot saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# PLOT 3 — Waterfall Plot (Single High-Risk User)
# ════════════════════════════════════════════════════════
def plot_waterfall_highrisk(shap_values, y_test, filename):
    print("\n📊 Generating SHAP Waterfall — High Risk User...")

    # Find the highest-risk churner in test set
    churn_indices = np.where(y_test.values == 1)[0]
    churn_shap    = shap_values.values[churn_indices]
    # Pick user with highest total positive SHAP (most convincingly at-risk)
    most_at_risk  = churn_indices[np.argmax(churn_shap.sum(axis=1))]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(COLORS["bg"])

    shap.waterfall_plot(
        shap_values[most_at_risk],
        max_display = 12,
        show        = False
    )

    plt.title(f"SHAP Waterfall — High Risk User (index {most_at_risk})\n"
              f"Why is this customer predicted to churn?",
              fontsize=12, fontweight="bold",
              color=COLORS["text"], pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}",
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.show()
    print(f"✅ Waterfall (high risk) saved → reports/figures/{filename}")
    print(f"   → User index  : {most_at_risk}")

    return most_at_risk


# ════════════════════════════════════════════════════════
# PLOT 4 — Waterfall Plot (Single Low-Risk User)
# ════════════════════════════════════════════════════════
def plot_waterfall_lowrisk(shap_values, y_test, filename):
    print("\n📊 Generating SHAP Waterfall — Low Risk User...")

    no_churn_indices = np.where(y_test.values == 0)[0]
    no_churn_shap    = shap_values.values[no_churn_indices]
    least_at_risk    = no_churn_indices[np.argmin(no_churn_shap.sum(axis=1))]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(COLORS["bg"])

    shap.waterfall_plot(
        shap_values[least_at_risk],
        max_display = 12,
        show        = False
    )

    plt.title(f"SHAP Waterfall — Low Risk User (index {least_at_risk})\n"
              f"Why is this customer predicted to stay?",
              fontsize=12, fontweight="bold",
              color=COLORS["text"], pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}",
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.show()
    print(f"✅ Waterfall (low risk) saved → reports/figures/{filename}")

    return least_at_risk


# ════════════════════════════════════════════════════════
# PLOT 5 — SHAP Dependence Plot (tenure vs churn)
# ════════════════════════════════════════════════════════
def plot_dependence(shap_values, X_test, filename):
    print("\n📊 Generating SHAP Dependence Plot (tenure)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    shap.dependence_plot(
        "tenure",
        shap_values.values,
        X_test,
        interaction_index = "MonthlyCharges",
        ax    = ax,
        show  = False,
        alpha = 0.6
    )

    ax.set_title("SHAP Dependence — Tenure × Monthly Charges",
                 fontsize=12, fontweight="bold",
                 color=COLORS["text"])
    ax.set_facecolor(COLORS["bg"])
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}",
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.show()
    print(f"✅ Dependence plot saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# GENERATE PER-USER SHAP EXPLANATION (for Agent Pipeline)
# ════════════════════════════════════════════════════════
def explain_single_user(explainer, user_row: pd.DataFrame,
                         feature_names: list) -> dict:
    """
    Called by the agent pipeline for each flagged at-risk user.
    Returns top 3 churn reasons as human-readable strings.

    Args:
        explainer    : fitted SHAP TreeExplainer
        user_row     : single-row DataFrame (1 × n_features)
        feature_names: list of feature column names

    Returns:
        dict with keys: shap_scores, top_reasons, risk_type
    """
    sv          = explainer(user_row)
    shap_vals   = sv.values[0]

    # Build ranked feature impact
    impact      = pd.Series(shap_vals, index=feature_names)
    top_positive = impact.nlargest(5)   # features pushing toward churn
    top_negative = impact.nsmallest(3)  # features pushing toward retention

    # Human-readable reason labels
    reason_map = {
        "tenure"             : "very short account tenure",
        "recency_score"      : "recently joined (high recency risk)",
        "MonthlyCharges"     : "high monthly charges",
        "is_monthly"         : "month-to-month contract (no commitment)",
        "is_fiber"           : "fiber optic internet (high churn segment)",
        "is_echeck"          : "electronic check payment method",
        "no_security"        : "no online security add-on",
        "charge_per_service" : "paying high price for few services",
        "active_services"    : "low number of active services",
        "lifetime_value"     : "low total lifetime spend",
        "Contract_Month-to-month": "month-to-month contract",
        "Contract_Two year"  : "long-term contract (retention factor)",
    }

    top_reasons = []
    for feat, val in top_positive.items():
        label = reason_map.get(feat, feat.replace("_", " "))
        top_reasons.append({
            "feature"   : feat,
            "shap_value": round(float(val), 4),
            "reason"    : label
        })

    # Classify risk type for agent routing
    has_support_issue = any(
        f in feature_names and
        impact.get(f, 0) > 0
        for f in ["no_security", "TechSupport", "OnlineSecurity"]
    )
    risk_type = "support_issue" if has_support_issue else "disengagement"

    return {
        "shap_scores"  : impact.to_dict(),
        "top_reasons"  : top_reasons[:3],
        "risk_type"    : risk_type,
        "top_positive" : top_positive.to_dict(),
        "top_negative" : top_negative.to_dict()
    }


# ════════════════════════════════════════════════════════
# SAVE EXPLAINER
# ════════════════════════════════════════════════════════
def save_explainer(explainer):
    joblib.dump(explainer, f"{OUTPUTS_DIR}/shap_explainer.pkl")
    print(f"\n✅ SHAP explainer saved → {OUTPUTS_DIR}/shap_explainer.pkl")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def run_shap():
    print("\n" + "="*55)
    print("   CHURN SENTINEL — SHAP Explainability")
    print("="*55 + "\n")

    # Load
    model, X_test, y_test, feature_names = load_artifacts()

    # Build explainer + compute SHAP values
    explainer, shap_values = build_shap_explainer(model, X_test)

    # Global plots
    plot_summary_beeswarm(shap_values, X_test,
                          "13_shap_beeswarm.png")
    plot_summary_bar(shap_values, X_test,
                     "14_shap_bar.png")

    # Individual user plots
    high_idx = plot_waterfall_highrisk(shap_values, y_test,
                                        "15_shap_waterfall_highrisk.png")
    low_idx  = plot_waterfall_lowrisk(shap_values, y_test,
                                       "16_shap_waterfall_lowrisk.png")

    # Dependence plot
    plot_dependence(shap_values, X_test,
                    "17_shap_dependence_tenure.png")

    # Demo: explain one high-risk user (for agent pipeline)
    print("\n🔍 Demo — Single User Explanation:")
    user_row    = X_test.iloc[[high_idx]]
    explanation = explain_single_user(explainer, user_row, feature_names)

    print(f"   Risk Type   : {explanation['risk_type']}")
    print(f"   Top Reasons :")
    for r in explanation["top_reasons"]:
        print(f"     • {r['reason']}"
              f"  (SHAP={r['shap_value']:+.4f})")

    # Save explainer for agent pipeline
    save_explainer(explainer)

    print("\n" + "="*55)
    print("✅ SHAP COMPLETE — 5 plots + explainer saved")
    print("="*55)

    return explainer, shap_values


if __name__ == "__main__":
    run_shap()