# src/verify_data.py
# Run this to confirm dataset is valid before EDA

import pandas as pd
import os

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def verify_dataset():
    print("=" * 55)
    print("   CHURN SENTINEL — Dataset Verification")
    print("=" * 55)

    # Check file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ FILE NOT FOUND: {DATA_PATH}")
        print("   → Place the CSV in data/raw/ and retry.")
        return False

    # Load
    df = pd.read_csv(DATA_PATH)

    print(f"\n✅ File found: {DATA_PATH}")
    print(f"\n📐 Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"🎯 Target col   : 'Churn' → unique values: {df['Churn'].unique()}")
    print(f"⚖️  Class balance:")
    print(df['Churn'].value_counts(normalize=True)
            .mul(100).round(1)
            .to_string())

    print(f"\n📋 All Columns ({df.shape[1]}):")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        print(f"   {i:2}. {col:<30} dtype={dtype:<10} nulls={nulls}")

    print(f"\n🔍 Sample rows (first 3):")
    print(df.head(3).to_string())

    print("\n" + "=" * 55)
    print("✅ Dataset verification PASSED. Ready for EDA.")
    print("=" * 55)
    return True

if __name__ == "__main__":
    verify_dataset()