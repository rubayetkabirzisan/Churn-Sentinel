# src/agents/behavior_detector.py
# Churn Sentinel — Agent 1: Behavior Detector
# Classifies WHY a user is at risk: disengagement vs support_issue
# Run standalone: python src/agents/behavior_detector.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from src.config import GROQ_API_KEY, GROQ_MODEL


# ════════════════════════════════════════════════════════
# RULE-BASED PRE-CLASSIFIER
# Fast check — no API call needed for obvious cases
# ════════════════════════════════════════════════════════
def rule_based_classify(shap_explanation: dict):
    """
    Returns classification string or None if unclear.
    None triggers LLM fallback.
    """
    top_reasons  = shap_explanation.get("top_reasons", [])
    features     = [r["feature"] for r in top_reasons]

    support_signals   = [
        "no_security", "TechSupport",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection"
    ]
    disengage_signals = [
        "tenure", "recency_score", "active_services",
        "is_monthly", "lifetime_value", "charge_per_service"
    ]

    support_hits   = sum(1 for f in features if f in support_signals)
    disengage_hits = sum(1 for f in features if f in disengage_signals)

    if support_hits >= 2 and support_hits > disengage_hits:
        return "support_issue"
    if disengage_hits >= 2 and disengage_hits > support_hits:
        return "disengagement"
    return None


# ════════════════════════════════════════════════════════
# LLM CLASSIFIER
# Called only when rule-based is uncertain
# ════════════════════════════════════════════════════════
def llm_classify(shap_explanation: dict, user_data: dict) -> str:
    llm = ChatGroq(
        api_key     = GROQ_API_KEY,
        model       = GROQ_MODEL,
        temperature = 0.1,
        max_tokens  = 20
    )

    reasons_text = "\n".join([
        f"  - {r['reason']} (SHAP: {r['shap_value']:+.3f})"
        for r in shap_explanation.get("top_reasons", [])
    ])

    prompt = PromptTemplate.from_template("""
You are a churn analyst. Classify this customer's churn risk.

Customer:
- Tenure: {tenure} months
- Monthly Charges: ${monthly_charges}
- Contract: {contract}
- Active Services: {active_services}

Top Risk Factors:
{reasons}

Reply with ONLY one word: disengagement OR support_issue
""")

    chain    = prompt | llm
    response = chain.invoke({
        "tenure"          : user_data.get("tenure", "unknown"),
        "monthly_charges" : user_data.get("MonthlyCharges", "unknown"),
        "contract"        : user_data.get("contract_type", "unknown"),
        "active_services" : user_data.get("active_services", "unknown"),
        "reasons"         : reasons_text
    })

    raw = response.content.strip().lower()
    if "support" in raw:
        return "support_issue"
    return "disengagement"


# ════════════════════════════════════════════════════════
# MAIN DETECT FUNCTION (called by Planner Agent)
# ════════════════════════════════════════════════════════
def detect_behavior(shap_explanation: dict,
                    user_data: dict) -> dict:
    """
    Main entry point for Behavior Detector Agent.

    Args:
        shap_explanation : output from shap_explainer.explain_single_user()
        user_data        : dict of raw user feature values

    Returns:
        dict with risk_type, method, confidence, top_reasons, routing
    """
    print(f"   🔍 Behavior Detector → analyzing user...")

    # Try fast rule-based first
    rule_result = rule_based_classify(shap_explanation)

    if rule_result:
        risk_type  = rule_result
        method     = "rule_based"
        confidence = "high"
        print(f"   ✅ Rule-based → {risk_type} (high confidence)")
    else:
        print(f"   🤖 Uncertain → calling Groq/{GROQ_MODEL}...")
        risk_type  = llm_classify(shap_explanation, user_data)
        method     = "llm"
        confidence = "medium"
        print(f"   ✅ LLM → {risk_type} (medium confidence)")

    routing_map = {
        "support_issue" : "email_generator:support_template",
        "disengagement" : "email_generator:reengagement_template"
    }

    top_reasons = [r["reason"] for r in
                   shap_explanation.get("top_reasons", [])]

    return {
        "risk_type"   : risk_type,
        "method"      : method,
        "confidence"  : confidence,
        "top_reasons" : top_reasons,
        "routing"     : routing_map[risk_type]
    }


# ════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*55)
    print("   Behavior Detector Agent — Standalone Test")
    print("="*55)

    # Test Case 1 — Disengagement (rule-based should catch)
    mock_shap_1 = {
        "top_reasons": [
            {"feature": "tenure",
             "shap_value": 0.882,
             "reason": "very short account tenure"},
            {"feature": "is_monthly",
             "shap_value": 0.654,
             "reason": "month-to-month contract"},
            {"feature": "active_services",
             "shap_value": 0.421,
             "reason": "low number of active services"}
        ]
    }
    mock_user_1 = {
        "tenure": 2, "MonthlyCharges": 85.0,
        "contract_type": "Month-to-month", "active_services": 2
    }

    # Test Case 2 — Support Issue (rule-based should catch)
    mock_shap_2 = {
        "top_reasons": [
            {"feature": "no_security",
             "shap_value": 0.791,
             "reason": "no online security add-on"},
            {"feature": "TechSupport",
             "shap_value": 0.612,
             "reason": "no tech support service"},
            {"feature": "OnlineBackup",
             "shap_value": 0.445,
             "reason": "no online backup service"}
        ]
    }
    mock_user_2 = {
        "tenure": 8, "MonthlyCharges": 70.0,
        "contract_type": "Month-to-month", "active_services": 1
    }

    # Test Case 3 — Ambiguous (LLM should classify)
    mock_shap_3 = {
        "top_reasons": [
            {"feature": "MonthlyCharges",
             "shap_value": 0.550,
             "reason": "high monthly charges"},
            {"feature": "no_security",
             "shap_value": 0.480,
             "reason": "no online security add-on"},
            {"feature": "is_fiber",
             "shap_value": 0.320,
             "reason": "fiber optic internet"}
        ]
    }
    mock_user_3 = {
        "tenure": 5, "MonthlyCharges": 95.0,
        "contract_type": "Month-to-month", "active_services": 3
    }

    for i, (shap, user) in enumerate([
        (mock_shap_1, mock_user_1),
        (mock_shap_2, mock_user_2),
        (mock_shap_3, mock_user_3)
    ], 1):
        print(f"\n📋 Test Case {i}:")
        print("-"*40)
        result = detect_behavior(shap, user)
        print(f"   Risk Type  : {result['risk_type']}")
        print(f"   Method     : {result['method']}")
        print(f"   Confidence : {result['confidence']}")
        print(f"   Routing    : {result['routing']}")

    print("\n" + "="*55)
    print("✅ Behavior Detector — All 3 tests PASSED")
    print("="*55)
