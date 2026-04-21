from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_CLEAN_DIR, DATA_RAW_PATH, ensure_directories


def clean_and_prepare_data() -> None:
    df = pd.read_csv(DATA_RAW_PATH)
    raw_shape = df.shape

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total_charges = int(df["TotalCharges"].isna().sum())

    df = df.dropna()

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df_encoded = pd.get_dummies(df, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop(columns=["Churn_Yes"])
    y = df_encoded["Churn_Yes"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    df_encoded.to_csv(DATA_CLEAN_DIR / "telco_clean.csv", index=False)
    X_train.to_csv(DATA_CLEAN_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_CLEAN_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_CLEAN_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_CLEAN_DIR / "y_test.csv", index=False)

    print("Data cleaning and preprocessing completed.")
    print(f"- Raw shape: {raw_shape}")
    print(f"- Missing TotalCharges removed: {missing_total_charges}")
    print(f"- Duplicate rows removed: {duplicate_count}")
    print(f"- Encoded shape: {df_encoded.shape}")
    print(f"- Train shape: {X_train.shape}, Test shape: {X_test.shape}")


if __name__ == "__main__":
    ensure_directories()
    clean_and_prepare_data()
