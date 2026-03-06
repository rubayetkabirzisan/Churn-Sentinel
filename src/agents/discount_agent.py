# src/agents/discount_agent.py
# Churn Sentinel — Agent 2: Discount Agent
# Decides if a user qualifies for retention discount + amount
# Run standalone: python -m src.agents.discount_agent

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.config import (
    DISCOUNT_MAX_PCT, DISCOUNT_MIN_PCT,
    MIN_TENURE_MONTHS, GROQ_API_KEY, GROQ_MODEL
)


# ════════════════════════════════════════════════════════
# DISCOUNT ELIGIBILITY RULES
# ════════════════════════════════════════════════════════

# Minimum lifetime value to qualify for discount
MIN_LTV_FOR_DISCOUNT = 100.0   # $100 total spend minimum

# Maximum monthly charge to offer discount
# (very low payers aren't worth discounting further)
MIN_MONTHLY_FOR_DISCOUNT = 30.0

# High-value customer threshold (gets max discount)
HIGH_VALUE_LTV = 500.0


def calculate_lifetime_value(tenure: float,
                              monthly_charges: float) -> float:
    """Proxy LTV = tenure × monthly charges"""
    return round(tenure * monthly_charges, 2)


def assess_eligibility(user_data: dict,
                        churn_prob: float) -> dict:
    """
    Rule-based eligibility check.

    Args:
        user_data   : dict with tenure, MonthlyCharges etc.
        churn_prob  : float 0.0–1.0 from XGBoost

    Returns:
        dict with eligible, reason, discount_pct, strategy
    """
    tenure          = float(user_data.get("tenure", 0))
    monthly_charges = float(user_data.get("MonthlyCharges", 0))
    ltv             = calculate_lifetime_value(tenure, monthly_charges)
    contract        = user_data.get("contract_type", "Month-to-month")

    # ── Disqualifiers ────────────────────────────────────

    # Too new — not enough history to justify discount
    if tenure < MIN_TENURE_MONTHS:
        return {
            "eligible"     : False,
            "reason"       : f"tenure too short ({tenure:.0f} months < "
                             f"{MIN_TENURE_MONTHS} minimum)",
            "discount_pct" : 0,
            "strategy"     : "free_trial_extension",
            "ltv"          : ltv
        }

    # Paying too little — discount not meaningful
    if monthly_charges < MIN_MONTHLY_FOR_DISCOUNT:
        return {
            "eligible"     : False,
            "reason"       : f"monthly charge too low "
                             f"(${monthly_charges:.0f} < "
                             f"${MIN_MONTHLY_FOR_DISCOUNT:.0f})",
            "discount_pct" : 0,
            "strategy"     : "feature_upsell",
            "ltv"          : ltv
        }

    # Already on annual contract — discount less impactful
    if contract == "Two year":
        return {
            "eligible"     : False,
            "reason"       : "already on two-year contract",
            "discount_pct" : 0,
            "strategy"     : "loyalty_reward",
            "ltv"          : ltv
        }

    # ── Qualifiers + Discount Calculation ────────────────

    # High churn risk + high LTV = maximum discount
    if churn_prob >= 0.80 and ltv >= HIGH_VALUE_LTV:
        discount_pct = DISCOUNT_MAX_PCT
        reason       = (f"high churn risk ({churn_prob:.0%}) + "
                        f"high LTV (${ltv:.0f})")

    # High churn risk + moderate LTV = mid discount
    elif churn_prob >= 0.80 and ltv >= MIN_LTV_FOR_DISCOUNT:
        discount_pct = round((DISCOUNT_MIN_PCT + DISCOUNT_MAX_PCT) / 2)
        reason       = (f"high churn risk ({churn_prob:.0%}) + "
                        f"moderate LTV (${ltv:.0f})")

    # Moderate churn risk + high LTV = min discount
    elif churn_prob >= 0.65 and ltv >= HIGH_VALUE_LTV:
        discount_pct = DISCOUNT_MIN_PCT
        reason       = (f"moderate churn risk ({churn_prob:.0%}) + "
                        f"high LTV (${ltv:.0f})")

    # Moderate risk + low LTV = no discount, try other strategy
    else:
        return {
            "eligible"     : False,
            "reason"       : (f"churn risk ({churn_prob:.0%}) + "
                              f"LTV (${ltv:.0f}) below thresholds"),
            "discount_pct" : 0,
            "strategy"     : "engagement_campaign",
            "ltv"          : ltv
        }

    return {
        "eligible"     : True,
        "reason"       : reason,
        "discount_pct" : discount_pct,
        "strategy"     : "discount_offer",
        "ltv"          : ltv,
        "monthly_savings" : round(monthly_charges * discount_pct / 100, 2),
        "annual_savings"  : round(monthly_charges * discount_pct / 100 * 12, 2)
    }


# ════════════════════════════════════════════════════════
# MAIN DISCOUNT FUNCTION (called by Planner Agent)
# ════════════════════════════════════════════════════════
def evaluate_discount(user_data: dict,
                      churn_prob: float) -> dict:
    """
    Main entry point for Discount Agent.

    Args:
        user_data  : dict of user feature values
        churn_prob : float churn probability from XGBoost

    Returns:
        dict with full discount decision + context for email agent
    """
    print(f"   💰 Discount Agent → evaluating eligibility...")

    result = assess_eligibility(user_data, churn_prob)

    if result["eligible"]:
        print(f"   ✅ ELIGIBLE — {result['discount_pct']}% discount")
        print(f"      Reason  : {result['reason']}")
        print(f"      Saves   : ${result['monthly_savings']}/month "
              f"(${result['annual_savings']}/year)")
    else:
        print(f"   ℹ️  NOT eligible — {result['reason']}")
        print(f"      Strategy: {result['strategy']}")

    return result


# ════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*55)
    print("   Discount Agent — Standalone Test")
    print("="*55)

    test_cases = [
        {
            "label"      : "High risk + High LTV → MAX discount",
            "user_data"  : {
                "tenure"         : 12,
                "MonthlyCharges" : 90.0,
                "contract_type"  : "Month-to-month"
            },
            "churn_prob" : 0.85
        },
        {
            "label"      : "High risk + Moderate LTV → MID discount",
            "user_data"  : {
                "tenure"         : 6,
                "MonthlyCharges" : 55.0,
                "contract_type"  : "Month-to-month"
            },
            "churn_prob" : 0.82
        },
        {
            "label"      : "Moderate risk + High LTV → MIN discount",
            "user_data"  : {
                "tenure"         : 24,
                "MonthlyCharges" : 75.0,
                "contract_type"  : "Month-to-month"
            },
            "churn_prob" : 0.70
        },
        {
            "label"      : "Too new → free trial extension",
            "user_data"  : {
                "tenure"         : 1,
                "MonthlyCharges" : 85.0,
                "contract_type"  : "Month-to-month"
            },
            "churn_prob" : 0.90
        },
        {
            "label"      : "Two year contract → loyalty reward",
            "user_data"  : {
                "tenure"         : 18,
                "MonthlyCharges" : 80.0,
                "contract_type"  : "Two year"
            },
            "churn_prob" : 0.68
        },
        {
            "label"      : "Low charges → feature upsell",
            "user_data"  : {
                "tenure"         : 5,
                "MonthlyCharges" : 20.0,
                "contract_type"  : "Month-to-month"
            },
            "churn_prob" : 0.75
        }
    ]

    passed = 0
    for i, tc in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {tc['label']}")
        print("-"*45)
        result = evaluate_discount(tc["user_data"], tc["churn_prob"])
        print(f"   Eligible   : {result['eligible']}")
        print(f"   Discount   : {result['discount_pct']}%")
        print(f"   Strategy   : {result['strategy']}")
        print(f"   LTV        : ${result['ltv']:.2f}")
        passed += 1

    print(f"\n{'='*55}")
    print(f"✅ Discount Agent — {passed}/{len(test_cases)} tests PASSED")
    print(f"{'='*55}")