import os
import pandas as pd
from utils import clean_credit_data, TARGET_COL


RAW_DATA_PATH = "data/raw/UCI_Credit_Card.csv"
PROCESSED_DATA_PATH = "data/processed/credit_clean.csv"


def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Please place UCI_Credit_Card.csv inside data/raw/"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("Raw dataset shape:", df.shape)
    print("\nRaw columns:")
    print(df.columns.tolist())

    cleaned_df = clean_credit_data(df)

    os.makedirs("data/processed", exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nCleaned dataset saved to:", PROCESSED_DATA_PATH)
    print("Cleaned dataset shape:", cleaned_df.shape)

    if TARGET_COL in cleaned_df.columns:
        print("\nDefault distribution:")
        print(cleaned_df[TARGET_COL].value_counts())
        print("\nDefault percentage:")
        print(cleaned_df[TARGET_COL].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()