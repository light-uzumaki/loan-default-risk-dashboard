import numpy as np
import pandas as pd


TARGET_COL = "is_default"


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "_", regex=False)
    )
    return df


def clean_credit_data(df):
    """
    Cleans the UCI Credit Card Default dataset.
    """

    df = clean_column_names(df)

    # Remove accidental unnamed index columns
    for col in ["unnamed:_0", "unnamed_0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Rename target column
    if "default_payment_next_month" in df.columns:
        df = df.rename(columns={"default_payment_next_month": TARGET_COL})

    if "default_next_month" in df.columns:
        df = df.rename(columns={"default_next_month": TARGET_COL})

    # Create ID if missing
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    numeric_cols = [
        "limit_bal", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"
    ]

    # Convert numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Target column
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    # Clean sex column
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({
            1: "male",
            2: "female",
            "1": "male",
            "2": "female",
            "male": "male",
            "female": "female"
        }).fillna("unknown")

    # Clean education column
    if "education" in df.columns:
        df["education"] = df["education"].replace({
            0: 4,
            5: 4,
            6: 4,
            "0": 4,
            "5": 4,
            "6": 4
        })

        df["education"] = df["education"].map({
            1: "graduate_school",
            2: "university",
            3: "high_school",
            4: "others",
            "1": "graduate_school",
            "2": "university",
            "3": "high_school",
            "4": "others"
        }).fillna("others")

    # Clean marriage column
    if "marriage" in df.columns:
        df["marriage"] = df["marriage"].replace({
            0: 3,
            "0": 3
        })

        df["marriage"] = df["marriage"].map({
            1: "married",
            2: "single",
            3: "others",
            "1": "married",
            "2": "single",
            "3": "others"
        }).fillna("others")

    bill_cols = [
        "bill_amt1", "bill_amt2", "bill_amt3",
        "bill_amt4", "bill_amt5", "bill_amt6"
    ]

    payment_cols = [
        "pay_amt1", "pay_amt2", "pay_amt3",
        "pay_amt4", "pay_amt5", "pay_amt6"
    ]

    delay_cols = [
        "pay_0", "pay_2", "pay_3",
        "pay_4", "pay_5", "pay_6"
    ]

    # Feature engineering
    existing_bill_cols = [col for col in bill_cols if col in df.columns]
    existing_payment_cols = [col for col in payment_cols if col in df.columns]
    existing_delay_cols = [col for col in delay_cols if col in df.columns]

    df["avg_bill_amt"] = df[existing_bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[existing_payment_cols].mean(axis=1)

    df["payment_ratio"] = np.where(
        df["avg_bill_amt"] > 0,
        df["avg_pay_amt"] / (df["avg_bill_amt"] + 1),
        0
    )

    df["months_with_delay"] = (df[existing_delay_cols] > 0).sum(axis=1)
    df["max_delay"] = df[existing_delay_cols].max(axis=1)
    df["avg_delay"] = df[existing_delay_cols].mean(axis=1)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=["21-30", "31-40", "41-50", "51-60", "60+"],
        include_lowest=True
    ).astype(str)

    df["limit_bucket"] = pd.cut(
        df["limit_bal"],
        bins=[0, 100000, 200000, 500000, np.inf],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True
    ).astype(str)

    return df