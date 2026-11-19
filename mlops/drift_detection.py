
# mlops/drift_detection.py

import numpy as np
import pandas as pd

from pathlib import Path
from src.data_loader import BASE_DIR
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.data_loader import load_data


# --------------------------------------------------
# 1. PSI CALCULATION
# --------------------------------------------------

def calculate_psi(expected, actual, bins=10):

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:

        raise ValueError(
            "Expected and actual data must contain values."
        )

    if len(np.unique(expected)) < 2:

        return 0.0

    breakpoints = np.unique(
        np.quantile(
            expected,
            np.linspace(0, 1, bins + 1)
        )
    )

    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints
    )

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero.
    epsilon = 0.0001

    expected_pct = np.clip(
        expected_pct, epsilon, None
    )

    actual_pct = np.clip(
        actual_pct, epsilon, None
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


# --------------------------------------------------
# 2. SENSOR DATA DRIFT MONITORING
# --------------------------------------------------

def detect_data_drift(reference_data, current_data):

    numeric_columns = reference_data.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numeric_columns:

        if column not in current_data.columns:
            continue

        psi_value = calculate_psi(
            reference_data[column],
            current_data[column]
        )

        if psi_value < 0.10:
            status = "STABLE"

        elif psi_value < 0.25:
            status = "MODERATE DRIFT"

        else:
            status = "SIGNIFICANT DRIFT"

        results.append({

            "feature": column,

            "psi": round(psi_value, 4),

            "status": status

        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 3. RUN DRIFT MONITORING
# --------------------------------------------------

if __name__ == "__main__":

    print("\nNEUROMAINTAIN DATA DRIFT MONITORING")

    # Reference data from original dataset.
    reference_df = clean_data(load_data())

    # Generate engineered features.
    reference_df = create_features(reference_df)

    # Current data from logged predictions.
    log_path = BASE_DIR / "mlops" / "prediction_logs.csv"

    if not log_path.exists():

        print("\nNo production logs available.")

        print("Run the Triage Agent first.")

    else:

        logs = pd.read_csv(log_path)

        if logs.empty:

            print("\nNo production records available.")

        else:

            sensor_records = pd.json_normalize(
                logs["sensor_data"].apply(
                    lambda x: __import__("json").loads(x)
                )
            )

            # Use only comparable raw numeric sensor features.
            sensor_columns = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]

            reference_df = reference_df.reindex(
                columns=sensor_columns
            )

            current_df = sensor_records.reindex(
                columns=sensor_columns
            )

            drift_report = detect_data_drift(
                reference_df,
                current_df
            )

            print("\nDrift Report:")

            print(
                drift_report.to_string(index=False)
            )
