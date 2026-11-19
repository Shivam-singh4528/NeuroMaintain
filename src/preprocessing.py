
import pandas as pd
from pathlib import Path

from src.data_loader import load_data, BASE_DIR


def clean_data(df):

    print("\nStarting data cleaning...")

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    # Remove unnecessary identifier columns
    df = df.drop(
        columns=["UDI", "Product ID"],
        errors="ignore"
    )

    # Handle missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Dataset is expected to have no missing values.
    # If missing values exist, don't silently delete rows.
    if df.isnull().any().any():
        raise ValueError(
            "Missing values found. Inspect them before proceeding."
        )

    print("\nCleaning completed!")
    print("Cleaned shape:", df.shape)

    return df


def save_clean_data(df):

    output_path = (
        BASE_DIR / "data" / "processed" / "cleaned_data.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nCleaned dataset saved at:")
    print(output_path)


if __name__ == "__main__":

    df = load_data()

    cleaned_df = clean_data(df)

    save_clean_data(cleaned_df)
