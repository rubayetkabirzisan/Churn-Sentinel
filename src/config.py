# src/config.py
# Churn Sentinel — Central Configuration
# Import this in every agent instead of hardcoding values

import os
from dotenv import load_dotenv
load_dotenv()

# ── Groq LLM ─────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.1-8b-instant"   # update here only if model changes

# ── Model paths ──────────────────────────────────────────
XGB_MODEL_PATH      = "outputs/xgb_model.pkl"
SCALER_PATH         = "outputs/scaler.pkl"
SHAP_EXPLAINER_PATH = "outputs/shap_explainer.pkl"
FEATURE_NAMES_PATH  = "outputs/feature_names.pkl"
BASELINE_MODEL_PATH = "outputs/baseline_model.pkl"

# ── Data paths ───────────────────────────────────────────
PROCESSED_DIR = "data/processed"
RAW_DIR       = "data/raw"
OUTPUTS_DIR   = "outputs"
FIGURES_DIR   = "reports/figures"

# ── Pipeline config ──────────────────────────────────────
CHURN_THRESHOLD   = 0.65     # flag users above this score
DISCOUNT_MAX_PCT  = 10       # max discount % to offer
DISCOUNT_MIN_PCT  = 5        # min discount % to offer
MIN_TENURE_MONTHS = 3        # minimum tenure to offer discount

# ── Email config ─────────────────────────────────────────
EMAIL_LOG_PATH        = "outputs/email_log.json"
EMAIL_SIMULATION_MODE = True   # True = log only, no real sending