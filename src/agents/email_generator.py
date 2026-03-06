# src/agents/email_generator.py
# Churn Sentinel — Agent 3: Email Generator
# Drafts personalized retention emails using Groq/llama3
# Run standalone: python -m src.agents.email_generator

import sys
import os
import json
import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from src.config import (
    GROQ_API_KEY, GROQ_MODEL,
    EMAIL_LOG_PATH, OUTPUTS_DIR
)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════
# EMAIL PROMPT TEMPLATES
# ════════════════════════════════════════════════════════

REENGAGEMENT_PROMPT = PromptTemplate.from_template("""
You are a warm, empathetic customer success manager at a SaaS company.
Write a personalized retention email for a customer showing signs of disengagement.

Customer Details:
- Tenure: {tenure} months with us
- Monthly Plan: ${monthly_charges}/month
- Contract: {contract_type}
- Active Services: {active_services}
- Risk Factors: {top_reasons}
{discount_section}

Email Requirements:
- Subject line first (prefix with "Subject: ")
- 3–4 short paragraphs
- Warm, personal tone — NOT robotic or generic
- Acknowledge their specific situation
- Highlight value they might be missing
- {cta_instruction}
- End with offer to schedule a call
- Max 200 words total

Write the email now:
""")

SUPPORT_PROMPT = PromptTemplate.from_template("""
You are a caring technical support manager at a SaaS company.
Write a personalized retention email for a customer facing service issues.

Customer Details:
- Tenure: {tenure} months with us
- Monthly Plan: ${monthly_charges}/month
- Contract: {contract_type}
- Service Gaps: {top_reasons}
{discount_section}

Email Requirements:
- Subject line first (prefix with "Subject: ")
- 3–4 short paragraphs
- Empathetic, solution-focused tone
- Acknowledge their service gaps directly
- Offer concrete solutions (security upgrade, tech support)
- {cta_instruction}
- Reassure them their data is protected
- Max 200 words total

Write the email now:
""")


# ════════════════════════════════════════════════════════
# BUILD PROMPT CONTEXT
# ════════════════════════════════════════════════════════
def build_prompt_context(user_data: dict,
                          behavior: dict,
                          discount: dict) -> dict:
    """Assembles all agent outputs into prompt variables."""

    # Format risk reasons as readable string
    top_reasons = behavior.get("top_reasons", [])
    reasons_str = ", ".join(top_reasons) if top_reasons \
                  else "general disengagement"

    # Build discount section for prompt
    if discount.get("eligible"):
        pct     = discount["discount_pct"]
        savings = discount.get("monthly_savings", 0)
        discount_section = (
            f"- Special Offer: {pct}% discount approved "
            f"(saves ${savings}/month)"
        )
        cta = (f"Include the {pct}% discount offer with "
               f"a clear call-to-action to claim it")
    else:
        strategy = discount.get("strategy", "engagement_campaign")
        strategy_map = {
            "free_trial_extension" : "Offer a 2-week free trial extension",
            "feature_upsell"       : "Highlight 2 premium features they aren't using",
            "loyalty_reward"       : "Mention an exclusive loyalty reward",
            "engagement_campaign"  : "Invite them to a free onboarding session"
        }
        discount_section = (
            f"- No discount (alternative: "
            f"{strategy_map.get(strategy, 'personalized offer')})"
        )
        cta = strategy_map.get(strategy,
                               "Include a clear next step call-to-action")

    return {
        "tenure"           : user_data.get("tenure", "a few"),
        "monthly_charges"  : user_data.get("MonthlyCharges", "N/A"),
        "contract_type"    : user_data.get("contract_type",
                                           "Month-to-month"),
        "active_services"  : user_data.get("active_services", "N/A"),
        "top_reasons"      : reasons_str,
        "discount_section" : discount_section,
        "cta_instruction"  : cta
    }


# ════════════════════════════════════════════════════════
# PARSE EMAIL OUTPUT
# ════════════════════════════════════════════════════════
def parse_email(raw_text: str) -> dict:
    """Splits LLM output into subject + body."""
    lines   = raw_text.strip().split("\n")
    subject = ""
    body    = []

    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
        else:
            body.append(line)

    body_text = "\n".join(body).strip()

    return {
        "subject" : subject or "We'd love to keep you with us",
        "body"    : body_text,
        "full"    : raw_text.strip()
    }


# ════════════════════════════════════════════════════════
# MAIN EMAIL GENERATOR (called by Planner Agent)
# ════════════════════════════════════════════════════════
def generate_email(user_id: str,
                   user_data: dict,
                   behavior: dict,
                   discount: dict,
                   churn_prob: float) -> dict:
    """
    Main entry point for Email Generator Agent.

    Args:
        user_id    : unique customer identifier
        user_data  : dict of user feature values
        behavior   : output from behavior_detector.detect_behavior()
        discount   : output from discount_agent.evaluate_discount()
        churn_prob : float churn probability from XGBoost

    Returns:
        dict with subject, body, metadata, ready to log/send
    """
    print(f"   ✉️  Email Generator → drafting email...")

    llm = ChatGroq(
        api_key     = GROQ_API_KEY,
        model       = GROQ_MODEL,
        temperature = 0.8,    # higher temp = more natural/varied emails
        max_tokens  = 400
    )

    # Pick correct template based on behavior classification
    risk_type = behavior.get("risk_type", "disengagement")
    prompt    = (SUPPORT_PROMPT if risk_type == "support_issue"
                 else REENGAGEMENT_PROMPT)

    # Build context
    context = build_prompt_context(user_data, behavior, discount)

    # Generate email
    chain    = prompt | llm
    response = chain.invoke(context)
    parsed   = parse_email(response.content)

    print(f"   ✅ Email drafted ({risk_type} template)")
    print(f"   📧 Subject: {parsed['subject']}")

    # Build full output record
    email_record = {
        "user_id"       : user_id,
        "timestamp"     : datetime.datetime.now().isoformat(),
        "churn_prob"    : round(churn_prob, 4),
        "risk_type"     : risk_type,
        "discount_pct"  : discount.get("discount_pct", 0),
        "eligible"      : discount.get("eligible", False),
        "subject"       : parsed["subject"],
        "body"          : parsed["body"],
        "top_reasons"   : behavior.get("top_reasons", []),
        "strategy"      : discount.get("strategy", "N/A"),
        "status"        : "simulated"   # change to "sent" with real SMTP
    }

    return email_record


# ════════════════════════════════════════════════════════
# LOG EMAIL TO JSON
# ════════════════════════════════════════════════════════
def log_email(email_record: dict):
    """Appends email record to the email log JSON file."""
    # Load existing log or start fresh
    if os.path.exists(EMAIL_LOG_PATH):
        with open(EMAIL_LOG_PATH, "r") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    else:
        log = []

    log.append(email_record)

    with open(EMAIL_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"   💾 Logged → {EMAIL_LOG_PATH} "
          f"(total: {len(log)} emails)")


# ════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*55)
    print("   Email Generator Agent — Standalone Test")
    print("="*55)

    # ── Test 1: Disengaged user WITH discount ────────────
    print("\n📋 Test 1: Disengaged user — discount eligible")
    print("-"*50)

    user_1 = {
        "tenure"          : 12,
        "MonthlyCharges"  : 90.0,
        "contract_type"   : "Month-to-month",
        "active_services" : 2
    }
    behavior_1 = {
        "risk_type"   : "disengagement",
        "top_reasons" : [
            "very short account tenure",
            "month-to-month contract",
            "low number of active services"
        ]
    }
    discount_1 = {
        "eligible"        : True,
        "discount_pct"    : 10,
        "monthly_savings" : 9.0,
        "annual_savings"  : 108.0,
        "strategy"        : "discount_offer"
    }

    record_1 = generate_email(
        user_id    = "USR_001",
        user_data  = user_1,
        behavior   = behavior_1,
        discount   = discount_1,
        churn_prob = 0.85
    )
    log_email(record_1)

    print(f"\n--- GENERATED EMAIL ---")
    print(f"Subject : {record_1['subject']}")
    print(f"Body    :\n{record_1['body']}")
    print(f"-----------------------")

    # ── Test 2: Support issue WITHOUT discount ───────────
    print("\n📋 Test 2: Support issue — no discount")
    print("-"*50)

    user_2 = {
        "tenure"          : 8,
        "MonthlyCharges"  : 70.0,
        "contract_type"   : "Month-to-month",
        "active_services" : 1
    }
    behavior_2 = {
        "risk_type"   : "support_issue",
        "top_reasons" : [
            "no online security add-on",
            "no tech support service",
            "no online backup service"
        ]
    }
    discount_2 = {
        "eligible"     : False,
        "discount_pct" : 0,
        "strategy"     : "feature_upsell"
    }

    record_2 = generate_email(
        user_id    = "USR_002",
        user_data  = user_2,
        behavior   = behavior_2,
        discount   = discount_2,
        churn_prob = 0.72
    )
    log_email(record_2)

    print(f"\n--- GENERATED EMAIL ---")
    print(f"Subject : {record_2['subject']}")
    print(f"Body    :\n{record_2['body']}")
    print(f"-----------------------")

    # ── Verify log file ──────────────────────────────────
    print(f"\n📂 Verifying email log...")
    with open(EMAIL_LOG_PATH, "r") as f:
        log = json.load(f)
    print(f"✅ Email log contains {len(log)} records")
    print(f"   → {EMAIL_LOG_PATH}")

    print(f"\n{'='*55}")
    print(f"✅ Email Generator — All tests PASSED")
    print(f"{'='*55}")