
import os
from pathlib import Path

import pandas as pd


def main():
    # Load tourism.csv, drop ID/index columns, rename target,
    # handle missing values, and save processed_tourism.csv.
    DATA_DIR = Path("tourism_project/data")
    RAW_PATH = DATA_DIR / "tourism.csv"
    PROCESSED_PATH = DATA_DIR / "processed_tourism.csv"

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} not found in repo.")

    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Drop unwanted columns
    cols_to_drop = ["Unnamed: 0", "CustomerID"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    print("Shape after dropping ID/index columns:", df.shape)

    # Rename target column
    if "ProdTaken" in df.columns:
        df = df.rename(columns={"ProdTaken": "will_purchase"})
    else:
        raise KeyError("Expected 'ProdTaken' column not found in dataset.")

    print("
Missing values before fill:")
    print(df.isna().sum())

    # Basic missing value handling
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    print("
Missing values after fill:")
    print(df.isna().sum())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Processed data saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
