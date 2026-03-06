# src/preprocessor.py
# Churn Sentinel — Data Preprocessing Pipeline
# Run: python src/preprocessor.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────
RAW_PATH       = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR  = "data/processed"
OUTPUTS_DIR    = "outputs"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR,   exist_ok=True)

# ── Constants ────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
TARGET_COL     = "Churn_Binary"
DROP_COLS      = ["customerID", "Churn"]


# ════════════════════════════════════════════════════════
# STEP 1 — Load Raw Data
# ════════════════════════════════════════════════════════
def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ════════════════════════════════════════════════════════
# STEP 2 — Basic Cleaning
# ════════════════════════════════════════════════════════
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix TotalCharges (whitespace → NaN → median fill)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_tc, inplace=True)

    # Encode binary target
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    # Strip whitespace from all object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"⚠️  Removed {before - after} duplicate rows")

    print(f"✅ Cleaning done — shape: {df.shape}")
    return df


# ════════════════════════════════════════════════════════
# STEP 3 — Feature Engineering (RFM + derived features)
# ════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── RFM Proxy Features ──────────────────────────────
    # Recency  → inverse of tenure (shorter tenure = more recent = higher risk)
    df["recency_score"]   = 1 / (df["tenure"] + 1)

    # Frequency → proxy: number of active services
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["active_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    # Monetary → monthly charges × tenure (lifetime spend proxy)
    df["lifetime_value"]  = df["MonthlyCharges"] * df["tenure"]

    # ── Behavioral Risk Flags ───────────────────────────
    # High monthly charge relative to services
    df["charge_per_service"] = df["MonthlyCharges"] / (df["active_services"] + 1)

    # Month-to-month contract flag (highest churn risk)
    df["is_monthly"]      = (df["Contract"] == "Month-to-month").astype(int)

    # Senior citizen flag (already numeric, rename for clarity)
    df["is_senior"]       = df["SeniorCitizen"]

    # No online security flag (common churn indicator)
    df["no_security"]     = (df["OnlineSecurity"] == "No").astype(int)

    # Fiber optic flag (associated with higher churn)
    df["is_fiber"]        = (df["InternetService"] == "Fiber optic").astype(int)

    # Electronic check payment (highest churn payment method)
    df["is_echeck"]       = (df["PaymentMethod"] == "Electronic check").astype(int)

    # Tenure bucket (new=0-12m, mid=13-36m, loyal=37m+)
    df["tenure_bucket"]   = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, df["tenure"].max()],
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(int)

    print(f"✅ Feature engineering done — new shape: {df.shape}")
    print(f"   → RFM features    : recency_score, active_services, lifetime_value")
    print(f"   → Risk flags      : is_monthly, is_senior, no_security, is_fiber, is_echeck")
    print(f"   → Derived metrics : charge_per_service, tenure_bucket")
    return df


# ════════════════════════════════════════════════════════
# STEP 4 — Encode Categorical Columns
# ════════════════════════════════════════════════════════
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary yes/no columns → 1/0
    binary_map = {"Yes": 1, "No": 0,
                  "No phone service": 0, "No internet service": 0}

    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # Multi-class categoricals → one-hot
    ohe_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Convert all bool columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"✅ Encoding done — shape after OHE: {df.shape}")
    return df


# ════════════════════════════════════════════════════════
# STEP 5 — Split + Scale
# ════════════════════════════════════════════════════════
def split_and_scale(df: pd.DataFrame):
    df = df.copy()

    # Drop non-feature columns
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop + [TARGET_COL])
    y = df[TARGET_COL]

    # Save feature names for SHAP later
    feature_names = X.columns.tolist()

    # Train/test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Back to DataFrame (keeps column names for SHAP)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_names)

    print(f"✅ Split done:")
    print(f"   → X_train : {X_train_scaled.shape}")
    print(f"   → X_test  : {X_test_scaled.shape}")
    print(f"   → y_train churn rate: {y_train.mean()*100:.1f}%")
    print(f"   → y_test  churn rate: {y_test.mean()*100:.1f}%")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ════════════════════════════════════════════════════════
# STEP 6 — Save Everything
# ════════════════════════════════════════════════════════
def save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_names):
    # Save processed CSVs
    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv",  index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv",  index=False)

    # Save scaler + feature names
    joblib.dump(scaler,        f"{OUTPUTS_DIR}/scaler.pkl")
    joblib.dump(feature_names, f"{OUTPUTS_DIR}/feature_names.pkl")

    print(f"\n✅ Saved to {PROCESSED_DIR}/:")
    print(f"   → X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"✅ Saved to {OUTPUTS_DIR}/:")
    print(f"   → scaler.pkl, feature_names.pkl")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def run_preprocessing():
    print("\n" + "=" * 55)
    print("   CHURN SENTINEL — Preprocessing Pipeline")
    print("=" * 55 + "\n")

    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    X_train, X_test, y_train, y_test, scaler, features = split_and_scale(df)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, features)

    print("\n" + "=" * 55)
    print("✅ PREPROCESSING COMPLETE")
    print(f"   Total features for modeling : {len(features)}")
    print(f"   Training samples            : {len(X_train):,}")
    print(f"   Test samples                : {len(X_test):,}")
    print("=" * 55)

    return X_train, X_test, y_train, y_test, scaler, features


if __name__ == "__main__":
    run_preprocessing()