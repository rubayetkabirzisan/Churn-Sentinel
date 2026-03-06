# src/model.py
# Churn Sentinel — Improved Model: XGBoost Classifier
# Run: python src/model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection  import GridSearchCV, StratifiedKFold
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)

# ── Paths ─────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"
FIGURES_DIR   = "reports/figures"
REPORT_PATH   = "reports/model_eval.md"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────
COLORS = {
    "churn"    : "#E63946",
    "no_churn" : "#2A9D8F",
    "accent"   : "#E9C46A",
    "bg"       : "#1A1A2E",
    "text"     : "#EAEAEA",
    "xgb"      : "#F4A261"
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
# LOAD DATA
# ════════════════════════════════════════════════════════
def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    print(f"✅ Data loaded — train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════
# COMPUTE CLASS IMBALANCE RATIO
# ════════════════════════════════════════════════════════
def get_scale_pos_weight(y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    ratio = round(neg / pos, 2)
    print(f"⚖️  Class ratio (neg/pos) = {ratio} → used as scale_pos_weight")
    return ratio


# ════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING (GridSearchCV)
# ════════════════════════════════════════════════════════
def tune_xgboost(X_train, y_train, scale_pos_weight):
    print("\n🔧 Running GridSearchCV (this takes ~2–3 min on CPU)...")

    param_grid = {
        "n_estimators"      : [100, 200],
        "max_depth"         : [3, 5],
        "learning_rate"     : [0.05, 0.1],
        "subsample"         : [0.8, 1.0],
        "colsample_bytree"  : [0.8, 1.0],
    }

    base_model = XGBClassifier(
        scale_pos_weight = scale_pos_weight,
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,
        tree_method      = "hist"       # fast CPU training
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        scoring    = "f1",              # optimize for F1 (imbalance-aware)
        cv         = cv,
        n_jobs     = -1,
        verbose    = 1,
        refit      = True
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best parameters found:")
    for k, v in grid_search.best_params_.items():
        print(f"   {k:<22}: {v}")
    print(f"   {'Best CV F1':<22}: {grid_search.best_score_*100:.2f}%")

    return grid_search.best_estimator_, grid_search.best_params_


# ════════════════════════════════════════════════════════
# EVALUATE
# ════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test, model_name="XGBoost"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model"     : model_name,
        "accuracy"  : round(accuracy_score(y_test,  y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test,    y_pred), 4),
        "f1"        : round(f1_score(y_test,        y_pred), 4),
        "roc_auc"   : round(roc_auc_score(y_test,  y_prob), 4),
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
def plot_confusion_matrix(y_test, y_pred, filename):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Confusion Matrix — XGBoost",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.RdYlGn)
    plt.colorbar(im, ax=ax)

    classes    = ["No Churn", "Churn"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

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
# PLOT 2 — ROC Curve (XGBoost vs Baseline overlay)
# ════════════════════════════════════════════════════════
def plot_roc_comparison(xgb_model, baseline_model,
                         X_test, y_test, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("ROC Curve — XGBoost vs Baseline",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    RocCurveDisplay.from_estimator(
        xgb_model, X_test, y_test, ax=ax,
        color=COLORS["xgb"], lw=2, name="XGBoost (improved)"
    )
    RocCurveDisplay.from_estimator(
        baseline_model, X_test, y_test, ax=ax,
        color=COLORS["churn"], lw=2,
        linestyle="--", name="Logistic Regression (baseline)"
    )
    ax.plot([0,1],[0,1], ":", color=COLORS["accent"],
            linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ ROC comparison saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# PLOT 3 — XGBoost Feature Importance
# ════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_names, filename):
    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Top 15 Feature Importances — XGBoost",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    colors_bar = plt.cm.YlOrRd(
        np.linspace(0.3, 0.9, len(importance)))
    ax.barh(importance.index, importance.values,
            color=colors_bar, edgecolor="#FFFFFF", linewidth=0.3)
    ax.set_xlabel("Importance Score (gain)")

    for i, (val, name) in enumerate(
            zip(importance.values, importance.index)):
        ax.text(val + 0.001, i, f"{val:.4f}",
                va="center", fontsize=9, color=COLORS["text"])

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Feature importance saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# PLOT 4 — Side-by-Side Metrics Comparison Bar Chart
# ════════════════════════════════════════════════════════
def plot_metrics_comparison(xgb_metrics, baseline_metrics, filename):
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels       = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    xgb_vals  = [xgb_metrics[m]      * 100 for m in metric_names]
    base_vals = [baseline_metrics[m] * 100 for m in metric_names]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("Model Comparison — XGBoost vs Logistic Regression",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    bars1 = ax.bar(x - width/2, base_vals, width,
                   label="Logistic Regression (baseline)",
                   color=COLORS["churn"], alpha=0.85,
                   edgecolor="#FFFFFF", linewidth=0.5)
    bars2 = ax.bar(x + width/2, xgb_vals, width,
                   label="XGBoost (improved)",
                   color=COLORS["xgb"], alpha=0.85,
                   edgecolor="#FFFFFF", linewidth=0.5)

    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(50, 100)
    ax.legend()
    ax.grid(axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom",
                fontsize=9, color=COLORS["text"])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom",
                fontsize=9, color=COLORS["text"])

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Comparison chart saved → reports/figures/{filename}")


# ════════════════════════════════════════════════════════
# UPDATE REPORT
# ════════════════════════════════════════════════════════
def update_report(xgb_metrics, baseline_metrics, best_params):
    with open(REPORT_PATH, "a") as f:
        f.write(f"""
## Improved Model: {xgb_metrics['model']}

### Best Hyperparameters
```
{chr(10).join(f"  {k}: {v}" for k, v in best_params.items())}
```

### Results

| Metric    | Logistic Regression | XGBoost | Improvement |
|-----------|--------------------:|--------:|------------:|
| Accuracy  | {baseline_metrics['accuracy']*100:.2f}% | {xgb_metrics['accuracy']*100:.2f}% | {(xgb_metrics['accuracy']-baseline_metrics['accuracy'])*100:+.2f}% |
| Precision | {baseline_metrics['precision']*100:.2f}% | {xgb_metrics['precision']*100:.2f}% | {(xgb_metrics['precision']-baseline_metrics['precision'])*100:+.2f}% |
| Recall    | {baseline_metrics['recall']*100:.2f}% | {xgb_metrics['recall']*100:.2f}% | {(xgb_metrics['recall']-baseline_metrics['recall'])*100:+.2f}% |
| F1-Score  | {baseline_metrics['f1']*100:.2f}% | {xgb_metrics['f1']*100:.2f}% | {(xgb_metrics['f1']-baseline_metrics['f1'])*100:+.2f}% |
| ROC-AUC   | {baseline_metrics['roc_auc']*100:.2f}% | {xgb_metrics['roc_auc']*100:.2f}% | {(xgb_metrics['roc_auc']-baseline_metrics['roc_auc'])*100:+.2f}% |

### Figures
- `reports/figures/09_cm_xgboost.png`
- `reports/figures/10_roc_comparison.png`
- `reports/figures/11_feature_importance_xgb.png`
- `reports/figures/12_metrics_comparison.png`

---
""")
    print(f"✅ Report updated → {REPORT_PATH}")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def run_xgboost():
    print("\n" + "="*55)
    print("   CHURN SENTINEL — XGBoost Model Training")
    print("="*55 + "\n")

    # Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    # Load baseline for comparison
    baseline_model   = joblib.load(f"{OUTPUTS_DIR}/baseline_model.pkl")
    baseline_metrics = joblib.load(f"{OUTPUTS_DIR}/baseline_metrics.pkl")

    # Class weight ratio
    scale_pos_weight = get_scale_pos_weight(y_train)

    # Tune + train
    best_model, best_params = tune_xgboost(
        X_train, y_train, scale_pos_weight)

    # Evaluate
    xgb_metrics, y_pred, y_prob = evaluate(
        best_model, X_test, y_test, "XGBoost")

    # Plots
    plot_confusion_matrix(
        y_test, y_pred, "09_cm_xgboost.png")

    plot_roc_comparison(
        best_model, baseline_model,
        X_test, y_test, "10_roc_comparison.png")

    plot_feature_importance(
        best_model, feature_names, "11_feature_importance_xgb.png")

    plot_metrics_comparison(
        xgb_metrics, baseline_metrics, "12_metrics_comparison.png")

    # Save
    joblib.dump(best_model,   f"{OUTPUTS_DIR}/xgb_model.pkl")
    joblib.dump(xgb_metrics,  f"{OUTPUTS_DIR}/xgb_metrics.pkl")
    joblib.dump(best_params,  f"{OUTPUTS_DIR}/xgb_best_params.pkl")
    update_report(xgb_metrics, baseline_metrics, best_params)

    # Final comparison printout
    print(f"\n{'='*55}")
    print("   FINAL COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<12} {'Baseline':>12} {'XGBoost':>12} {'Delta':>10}")
    print(f"  {'-'*48}")
    for m in ["accuracy","precision","recall","f1","roc_auc"]:
        delta = (xgb_metrics[m] - baseline_metrics[m]) * 100
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {m.capitalize():<12}"
              f" {baseline_metrics[m]*100:>11.2f}%"
              f" {xgb_metrics[m]*100:>11.2f}%"
              f" {arrow}{abs(delta):>8.2f}%")

    print(f"\n✅ XGBoost model saved → outputs/xgb_model.pkl")
    print("="*55)

    return best_model, xgb_metrics


if __name__ == "__main__":
    run_xgboost()