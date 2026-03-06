# src/agents/planner.py
# Churn Sentinel — Agent 4: Planner (Orchestrator)
# Connects SHAP → Behavior → Discount → Email for one user
# Run standalone: python -m src.agents.planner

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
       os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import joblib
import pandas as pd
from src.config import (
    XGB_MODEL_PATH, SHAP_EXPLAINER_PATH,
    FEATURE_NAMES_PATH, CHURN_THRESHOLD
)
from src.shap_explainer        import explain_single_user
from src.agents.behavior_detector import detect_behavior
from src.agents.discount_agent    import evaluate_discount
from src.agents.email_generator   import generate_email, log_email


# ════════════════════════════════════════════════════════
# LOAD SHARED ARTIFACTS (loaded once at import)
# ════════════════════════════════════════════════════════
print("🔧 Loading model artifacts...")
xgb_model     = joblib.load(XGB_MODEL_PATH)
shap_explainer = joblib.load(SHAP_EXPLAINER_PATH)
feature_names  = joblib.load(FEATURE_NAMES_PATH)
print("✅ Artifacts loaded\n")


# ════════════════════════════════════════════════════════
# SINGLE USER PIPELINE
# ════════════════════════════════════════════════════════
def run_for_user(user_id: str,
                 user_features: pd.DataFrame,
                 user_raw: dict) -> dict:
    """
    Full agentic pipeline for ONE at-risk user.

    Args:
        user_id       : unique customer ID string
        user_features : preprocessed single-row DataFrame
                        (same format as X_train)
        user_raw      : raw user dict with readable values
                        (tenure, MonthlyCharges, contract_type etc.)

    Returns:
        dict with full pipeline results
    """
    print(f"\n{'='*55}")
    print(f"  PLANNER — Processing User: {user_id}")
    print(f"{'='*55}")

    # ── Step 1: Get churn probability ────────────────────
    churn_prob = float(
        xgb_model.predict_proba(user_features)[:, 1][0]
    )
    print(f"\n  📊 Step 1 — Churn Probability: {churn_prob:.1%}")

    # Check threshold
    if churn_prob < CHURN_THRESHOLD:
        print(f"  ✅ Below threshold ({CHURN_THRESHOLD}) — no action needed")
        return {
            "user_id"    : user_id,
            "churn_prob" : churn_prob,
            "action"     : "no_action",
            "reason"     : f"churn_prob {churn_prob:.1%} < "
                           f"threshold {CHURN_THRESHOLD}"
        }

    print(f"  ⚠️  Above threshold ({CHURN_THRESHOLD}) — FLAGGED")

    # ── Step 2: SHAP explanation ──────────────────────────
    print(f"\n  🔍 Step 2 — SHAP Explanation:")
    shap_explanation = explain_single_user(
        shap_explainer, user_features, feature_names
    )
    print(f"  Top reasons:")
    for r in shap_explanation["top_reasons"]:
        print(f"    • {r['reason']} (SHAP={r['shap_value']:+.3f})")

    # ── Step 3: Behavior Detection ────────────────────────
    print(f"\n  🤖 Step 3 — Behavior Detection:")
    behavior = detect_behavior(shap_explanation, user_raw)

    # ── Step 4: Discount Evaluation ───────────────────────
    print(f"\n  💰 Step 4 — Discount Evaluation:")
    discount = evaluate_discount(user_raw, churn_prob)

    # ── Step 5: Email Generation ──────────────────────────
    print(f"\n  ✉️  Step 5 — Email Generation:")
    email_record = generate_email(
        user_id    = user_id,
        user_data  = user_raw,
        behavior   = behavior,
        discount   = discount,
        churn_prob = churn_prob
    )

    # ── Step 6: Log ───────────────────────────────────────
    log_email(email_record)

    # ── Summary ───────────────────────────────────────────
    result = {
        "user_id"      : user_id,
        "churn_prob"   : round(churn_prob, 4),
        "risk_type"    : behavior["risk_type"],
        "confidence"   : behavior["confidence"],
        "discount_pct" : discount.get("discount_pct", 0),
        "eligible"     : discount.get("eligible", False),
        "email_subject": email_record["subject"],
        "strategy"     : discount.get("strategy", "N/A"),
        "top_reasons"  : behavior["top_reasons"],
        "action"       : "email_sent"
    }

    print(f"\n  {'─'*50}")
    print(f"  ✅ PIPELINE COMPLETE for {user_id}")
    print(f"     Churn Prob  : {churn_prob:.1%}")
    print(f"     Risk Type   : {behavior['risk_type']}")
    print(f"     Discount    : {discount.get('discount_pct',0)}%")
    print(f"     Email       : {email_record['subject']}")
    print(f"  {'─'*50}")

    return result


# ════════════════════════════════════════════════════════
# BATCH PIPELINE (multiple users)
# ════════════════════════════════════════════════════════
def run_batch(users_df: pd.DataFrame,
              users_raw: list[dict],
              user_ids: list[str]) -> list[dict]:
    """
    Runs planner for a batch of users.

    Args:
        users_df  : preprocessed DataFrame (n_users × n_features)
        users_raw : list of raw user dicts
        user_ids  : list of user ID strings

    Returns:
        list of result dicts
    """
    results  = []
    flagged  = 0
    skipped  = 0

    print(f"\n🚀 Starting batch pipeline for "
          f"{len(user_ids)} users...")
    print(f"   Threshold: {CHURN_THRESHOLD}")

    for i, (uid, raw) in enumerate(zip(user_ids, users_raw)):
        features = users_df.iloc[[i]]
        result   = run_for_user(uid, features, raw)
        results.append(result)

        if result.get("action") == "email_sent":
            flagged += 1
        else:
            skipped += 1

    print(f"\n{'='*55}")
    print(f"  BATCH COMPLETE")
    print(f"  Total users  : {len(user_ids)}")
    print(f"  Flagged+sent : {flagged}")
    print(f"  Below thresh : {skipped}")
    print(f"{'='*55}")

    return results


# ════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*55)
    print("   Planner Agent — Standalone Test (3 mock users)")
    print("="*55)

    # Load test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Pick 3 real users from test set:
    # 1 high-risk churner, 1 low-risk, 1 mid-risk
    churn_probs = xgb_model.predict_proba(X_test)[:, 1]

    import numpy as np
    churn_idx  = np.where(y_test.values == 1)[0]
    high_idx   = churn_idx[np.argmax(churn_probs[churn_idx])]
    low_idx    = np.argmin(churn_probs)
    mid_idx    = churn_idx[
        np.argmin(np.abs(churn_probs[churn_idx] - 0.70))
    ]

    # Build test users
    test_users = [
        {
            "idx"     : high_idx,
            "id"      : f"USR_HIGH_{high_idx}",
            "label"   : "High Risk Churner",
            "raw"     : {
                "tenure"          : int(X_test.iloc[high_idx]["tenure"]),
                "MonthlyCharges"  : float(X_test.iloc[high_idx]["MonthlyCharges"]),
                "contract_type"   : "Month-to-month",
                "active_services" : int(X_test.iloc[high_idx].get(
                                    "active_services", 2))
            }
        },
        {
            "idx"     : low_idx,
            "id"      : f"USR_LOW_{low_idx}",
            "label"   : "Low Risk (should be skipped)",
            "raw"     : {
                "tenure"          : int(X_test.iloc[low_idx]["tenure"]),
                "MonthlyCharges"  : float(X_test.iloc[low_idx]["MonthlyCharges"]),
                "contract_type"   : "Two year",
                "active_services" : int(X_test.iloc[low_idx].get(
                                    "active_services", 6))
            }
        },
        {
            "idx"     : mid_idx,
            "id"      : f"USR_MID_{mid_idx}",
            "label"   : "Mid Risk Churner",
            "raw"     : {
                "tenure"          : int(X_test.iloc[mid_idx]["tenure"]),
                "MonthlyCharges"  : float(X_test.iloc[mid_idx]["MonthlyCharges"]),
                "contract_type"   : "Month-to-month",
                "active_services" : int(X_test.iloc[mid_idx].get(
                                    "active_services", 3))
            }
        }
    ]

    # Run each user through full pipeline
    all_results = []
    for tu in test_users:
        print(f"\n{'*'*55}")
        print(f"  TEST USER: {tu['label']}")
        print(f"{'*'*55}")
        features = X_test.iloc[[tu["idx"]]]
        result   = run_for_user(tu["id"], features, tu["raw"])
        all_results.append(result)

    # Final summary table
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'User':<20} {'Prob':>6} {'Risk':<16} {'Disc':>5} {'Action'}")
    print(f"  {'─'*52}")
    for r in all_results:
        print(f"  {r['user_id']:<20}"
              f" {r['churn_prob']:>5.1%}"
              f" {r.get('risk_type','N/A'):<16}"
              f" {r.get('discount_pct',0):>4}%"
              f" {r.get('action','N/A')}")

    print(f"\n✅ Planner Agent — Standalone Test COMPLETE")
    print(f"   Check outputs/email_log.json for logged emails")
    print(f"{'='*55}")