
import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "raw" / "ai4i2020.csv"


def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    return df


if __name__ == "__main__":

    df = load_data()

    print("\nDataset loaded successfully!")
