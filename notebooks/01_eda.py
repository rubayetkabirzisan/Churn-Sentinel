# notebooks/01_eda.py
# Churn Sentinel — Exploratory Data Analysis
# Run: python notebooks/01_eda.py
# OR: copy cells into Jupyter notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────
DATA_PATH    = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIGURES_PATH = "reports/figures"
os.makedirs(FIGURES_PATH, exist_ok=True)

COLORS = {
    "churn"    : "#E63946",
    "no_churn" : "#2A9D8F",
    "accent"   : "#E9C46A",
    "bg"       : "#1A1A2E",
    "text"     : "#EAEAEA"
}

plt.rcParams.update({
    "figure.facecolor"  : COLORS["bg"],
    "axes.facecolor"    : COLORS["bg"],
    "axes.edgecolor"    : COLORS["text"],
    "axes.labelcolor"   : COLORS["text"],
    "xtick.color"       : COLORS["text"],
    "ytick.color"       : COLORS["text"],
    "text.color"        : COLORS["text"],
    "font.family"       : "monospace",
    "grid.color"        : "#2E2E4E",
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5
})

print("=" * 55)
print("   CHURN SENTINEL — EDA Report")
print("=" * 55)

# ── Load Data ────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# Fix TotalCharges (has spaces → NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode target
df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

print(f"🔧 TotalCharges fixed (coerced + median-filled)")
print(f"🎯 Churn rate: {df['Churn_Binary'].mean()*100:.1f}%")


# ════════════════════════════════════════════════════════
# PLOT 1 — Class Distribution (Pie + Bar side by side)
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PLOT 1 — Churn Class Distribution",
             fontsize=14, fontweight="bold", color=COLORS["text"])

churn_counts = df["Churn"].value_counts()

# Bar
axes[0].bar(churn_counts.index,
            churn_counts.values,
            color=[COLORS["no_churn"], COLORS["churn"]],
            edgecolor="#FFFFFF", linewidth=0.5, width=0.5)
axes[0].set_title("Count", color=COLORS["text"])
axes[0].set_ylabel("Number of Customers")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, f"{v:,}", ha="center",
                 color=COLORS["text"], fontweight="bold")

# Pie
axes[1].pie(
    churn_counts.values,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    colors=[COLORS["no_churn"], COLORS["churn"]],
    startangle=90,
    wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2}
)
axes[1].set_title("Proportion", color=COLORS["text"])

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/01_class_distribution.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot 1 saved → reports/figures/01_class_distribution.png")


# ════════════════════════════════════════════════════════
# PLOT 2 — Churn by Contract Type, Internet, Payment
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PLOT 2 — Churn Rate by Key Categorical Features",
             fontsize=14, fontweight="bold", color=COLORS["text"])

cat_features = ["Contract", "InternetService", "PaymentMethod"]

for ax, feat in zip(axes, cat_features):
    churn_rate = df.groupby(feat)["Churn_Binary"].mean().sort_values(ascending=False)
    bars = ax.barh(churn_rate.index, churn_rate.values * 100,
                   color=COLORS["churn"], edgecolor="#FFFFFF",
                   linewidth=0.5, alpha=0.85)
    ax.set_title(feat, color=COLORS["text"], fontweight="bold")
    ax.set_xlabel("Churn Rate (%)")
    ax.axvline(x=df["Churn_Binary"].mean() * 100,
               color=COLORS["accent"], linestyle="--",
               linewidth=1.5, label="Avg churn rate")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, churn_rate.values):
        ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center",
                color=COLORS["text"], fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/02_churn_by_category.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot 2 saved → reports/figures/02_churn_by_category.png")


# ════════════════════════════════════════════════════════
# PLOT 3 — Numerical Feature Distributions (Churn vs No)
# ════════════════════════════════════════════════════════
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PLOT 3 — Numerical Features: Churn vs No Churn",
             fontsize=14, fontweight="bold", color=COLORS["text"])

for ax, feat in zip(axes, num_features):
    for label, color in [("No", COLORS["no_churn"]), ("Yes", COLORS["churn"])]:
        subset = df[df["Churn"] == label][feat]
        ax.hist(subset, bins=30, alpha=0.65, color=color,
                label=f"Churn={label}", edgecolor="none", density=True)
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/03_numerical_distributions.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot 3 saved → reports/figures/03_numerical_distributions.png")


# ════════════════════════════════════════════════════════
# PLOT 4 — Correlation Heatmap (numerical only)
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("PLOT 4 — Correlation Heatmap",
             fontsize=14, fontweight="bold", color=COLORS["text"])

num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges",
            "TotalCharges", "Churn_Binary"]
corr = df[num_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, ax=ax,
    annot=True, fmt=".2f",
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    linewidths=0.5, linecolor=COLORS["bg"],
    annot_kws={"size": 11, "weight": "bold"}
)
ax.set_title("Feature Correlations", color=COLORS["text"])

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/04_correlation_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot 4 saved → reports/figures/04_correlation_heatmap.png")


# ════════════════════════════════════════════════════════
# PLOT 5 — Tenure vs Monthly Charges (Scatter by Churn)
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("PLOT 5 — Tenure vs Monthly Charges",
             fontsize=14, fontweight="bold", color=COLORS["text"])

for label, color, marker in [
    ("No",  COLORS["no_churn"], "o"),
    ("Yes", COLORS["churn"],    "X")
]:
    subset = df[df["Churn"] == label]
    ax.scatter(subset["tenure"], subset["MonthlyCharges"],
               c=color, alpha=0.4, s=15,
               marker=marker, label=f"Churn={label}")

ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Monthly Charges ($)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/05_tenure_vs_charges_scatter.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot 5 saved → reports/figures/05_tenure_vs_charges_scatter.png")


# ════════════════════════════════════════════════════════
# SUMMARY STATS PRINTOUT
# ════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("   EDA SUMMARY — Key Findings")
print("=" * 55)
print(f"""
📌 Dataset size       : {df.shape[0]:,} customers
📌 Features           : {df.shape[1]-2} input + 1 target
📌 Churn rate         : {df['Churn_Binary'].mean()*100:.1f}% (class imbalance present)
📌 Avg tenure churned : {df[df['Churn']=='Yes']['tenure'].mean():.1f} months
📌 Avg tenure kept    : {df[df['Churn']=='No']['tenure'].mean():.1f} months
📌 Top churn contract : {df.groupby('Contract')['Churn_Binary'].mean().idxmax()}
📌 Highest risk group : Month-to-month + Fiber optic + Electronic check
📌 Null values fixed  : TotalCharges (median imputation)
📌 Figures saved to   : reports/figures/ (5 plots)
""")
print("✅ EDA COMPLETE — Ready for preprocessing (M1.T4)")
print("=" * 55)