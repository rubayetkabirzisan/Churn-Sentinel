# src/pipeline.py
# Churn Sentinel — End-to-End Pipeline Runner
# Run: python src/pipeline.py
# Run fast test: python src/pipeline.py --test

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
import joblib
import argparse
import numpy as np
import pandas as pd
import datetime
from src.config import (
    XGB_MODEL_PATH, SHAP_EXPLAINER_PATH,
    FEATURE_NAMES_PATH, SCALER_PATH,
    CHURN_THRESHOLD, EMAIL_LOG_PATH,
    PROCESSED_DIR, OUTPUTS_DIR
)
from src.agents.planner import run_for_user


# ════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ════════════════════════════════════════════════════════
def load_artifacts():
    print("🔧 Loading pipeline artifacts...")
    artifacts = {
        "xgb_model"      : joblib.load(XGB_MODEL_PATH),
        "shap_explainer" : joblib.load(SHAP_EXPLAINER_PATH),
        "feature_names"  : joblib.load(FEATURE_NAMES_PATH),
        "scaler"         : joblib.load(SCALER_PATH),
    }
    print("✅ All artifacts loaded\n")
    return artifacts


# ════════════════════════════════════════════════════════
# LOAD + PREPARE DATA
# ════════════════════════════════════════════════════════
def load_data(test_mode: bool = False,
              test_size: int = 10) -> tuple:
    """
    Loads processed test set.
    In test_mode, uses only first N users for speed.
    """
    X      = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y      = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    raw_df = pd.read_csv(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    # Fix raw data types
    raw_df["TotalCharges"] = pd.to_numeric(
        raw_df["TotalCharges"], errors="coerce"
    ).fillna(0)

    if test_mode:
        X      = X.iloc[:test_size]
        y      = y.iloc[:test_size]
        raw_df = raw_df.iloc[:test_size]
        print(f"⚡ TEST MODE — using first {test_size} users only")
    else:
        print(f"📦 FULL MODE — processing {len(X):,} users")

    print(f"   Features : {X.shape[1]}")
    print(f"   Samples  : {len(X):,}\n")

    return X, y, raw_df


# ════════════════════════════════════════════════════════
# SCORE ALL USERS
# ════════════════════════════════════════════════════════
def score_users(model, X: pd.DataFrame) -> np.ndarray:
    """Returns churn probability array for all users."""
    probs = model.predict_proba(X)[:, 1]
    print(f"📊 Scoring complete:")
    print(f"   Users scored        : {len(probs):,}")
    print(f"   Above threshold     : "
          f"{(probs >= CHURN_THRESHOLD).sum():,} "
          f"({(probs >= CHURN_THRESHOLD).mean()*100:.1f}%)")
    print(f"   Avg churn prob      : {probs.mean()*100:.1f}%")
    print(f"   Max churn prob      : {probs.max()*100:.1f}%\n")
    return probs


# ════════════════════════════════════════════════════════
# BUILD RAW USER DICT (for agents)
# ════════════════════════════════════════════════════════
def build_raw_user(raw_row, features_row) -> dict:
    """
    Builds readable user dict for agent consumption.
    Tries raw CSV first, falls back to feature values.
    """
    # Contract type mapping from one-hot
    contract = "Month-to-month"
    if features_row.get("Contract_Two year", 0) == 1:
        contract = "Two year"
    elif features_row.get("Contract_One year", 0) == 1:
        contract = "One year"

    return {
        "tenure"          : float(raw_row.get("tenure", 0)),
        "MonthlyCharges"  : float(raw_row.get("MonthlyCharges", 0)),
        "TotalCharges"    : float(raw_row.get("TotalCharges", 0)),
        "contract_type"   : contract,
        "active_services" : float(
            features_row.get("active_services", 0)
        ),
        "is_monthly"      : int(features_row.get("is_monthly", 0)),
        "is_fiber"        : int(features_row.get("is_fiber", 0)),
        "SeniorCitizen"   : int(raw_row.get("SeniorCitizen", 0)),
    }


# ════════════════════════════════════════════════════════
# SAVE PIPELINE REPORT
# ════════════════════════════════════════════════════════
def save_pipeline_report(results: list,
                          probs: np.ndarray,
                          duration: float,
                          test_mode: bool):
    """Saves a markdown summary report of the pipeline run."""
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/pipeline_run.md"

    flagged  = [r for r in results if r.get("action") == "email_sent"]
    skipped  = [r for r in results if r.get("action") == "no_action"]

    disc_eligible = [r for r in flagged if r.get("eligible")]
    support_ct    = sum(1 for r in flagged
                        if r.get("risk_type") == "support_issue")
    disengage_ct  = sum(1 for r in flagged
                        if r.get("risk_type") == "disengagement")

    report = f"""# Churn Sentinel — Pipeline Run Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: {'TEST' if test_mode else 'FULL'}

## Summary

| Metric | Value |
|--------|-------|
| Total users processed | {len(results):,} |
| Flagged (above threshold) | {len(flagged):,} |
| Below threshold (no action) | {len(skipped):,} |
| Emails generated | {len(flagged):,} |
| Discount eligible | {len(disc_eligible):,} |
| Avg churn probability | {probs.mean()*100:.1f}% |
| Pipeline duration | {duration:.1f}s |

## Risk Type Breakdown

| Risk Type | Count | % of Flagged |
|-----------|-------|-------------|
| Disengagement | {disengage_ct} | {disengage_ct/max(len(flagged),1)*100:.1f}% |
| Support Issue | {support_ct} | {support_ct/max(len(flagged),1)*100:.1f}% |

## Flagged Users

| User ID | Churn Prob | Risk Type | Discount | Subject |
|---------|-----------|-----------|----------|---------|
"""
    for r in flagged[:20]:   # show max 20
        report += (
            f"| {r['user_id']} "
            f"| {r['churn_prob']*100:.1f}% "
            f"| {r.get('risk_type','N/A')} "
            f"| {r.get('discount_pct',0)}% "
            f"| {r.get('email_subject','N/A')[:40]} |\n"
        )

    report += f"""
## Output Files
- Email log : `{EMAIL_LOG_PATH}`
- This report : `{report_path}`

---
*Churn Sentinel v1.0 — XGBoost + Multi-Agent AI*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Pipeline report saved → {report_path}")
    return report_path


# ════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════
def run_pipeline(test_mode: bool = False,
                 test_size: int  = 10):
    start_time = datetime.datetime.now()

    print("\n" + "="*55)
    print("   CHURN SENTINEL — Pipeline Runner")
    print(f"   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55 + "\n")

    # Load
    artifacts      = load_artifacts()
    X, y, raw_df   = load_data(test_mode, test_size)
    model          = artifacts["xgb_model"]

    # Score all users
    print("📊 Scoring all users...")
    probs = score_users(model, X)

    # Clear old email log for fresh run
    if os.path.exists(EMAIL_LOG_PATH):
        os.remove(EMAIL_LOG_PATH)
        print(f"🗑️  Cleared old email log\n")

    # Process each user through agent pipeline
    results   = []
    processed = 0
    flagged   = 0

    print(f"🤖 Starting agent pipeline...\n")

    for i in range(len(X)):
        user_id      = f"USR_{i:04d}"
        churn_prob   = float(probs[i])
        user_features = X.iloc[[i]]

        # Build raw user dict
        raw_row      = raw_df.iloc[i].to_dict()
        features_row = X.iloc[i].to_dict()
        user_raw     = build_raw_user(raw_row, features_row)

        # Run through planner
        result = run_for_user(
            user_id       = user_id,
            user_features = user_features,
            user_raw      = user_raw
        )
        results.append(result)
        processed += 1

        if result.get("action") == "email_sent":
            flagged += 1

        # Progress update every 5 users
        if (i + 1) % 5 == 0:
            print(f"\n   ⏳ Progress: {i+1}/{len(X)} users "
                  f"| Flagged so far: {flagged}\n")

    # Duration
    duration = (datetime.datetime.now() - start_time).total_seconds()

    # Save report
    print(f"\n{'='*55}")
    print(f"  SAVING REPORTS...")
    print(f"{'='*55}")
    save_pipeline_report(results, probs, duration, test_mode)

    # Final summary
    print(f"\n{'='*55}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"{'='*55}")
    print(f"  Total processed  : {processed:,}")
    print(f"  Flagged + emailed: {flagged:,}")
    print(f"  Skipped          : {processed - flagged:,}")
    print(f"  Duration         : {duration:.1f}s")
    print(f"  Email log        : {EMAIL_LOG_PATH}")
    print(f"  Report           : reports/pipeline_run.md")
    print(f"{'='*55}\n")

    return results


# ════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Churn Sentinel Pipeline Runner"
    )
    parser.add_argument(
        "--test",
        action  = "store_true",
        help    = "Run in test mode (first 10 users only)"
    )
    parser.add_argument(
        "--size",
        type    = int,
        default = 10,
        help    = "Number of users in test mode (default: 10)"
    )
    args = parser.parse_args()

    run_pipeline(
        test_mode = args.test,
        test_size = args.size
    )