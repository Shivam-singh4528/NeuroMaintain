
# mlops/monitoring.py

import json
import pandas as pd

from pathlib import Path
from datetime import datetime
from src.data_loader import BASE_DIR


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

MLOPS_DIR = BASE_DIR / "mlops"

LOG_PATH = MLOPS_DIR / "prediction_logs.csv"

MLOPS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOG MACHINE PREDICTIONS
# --------------------------------------------------

def log_prediction(
    sensor_data,
    triage_result,
    inference_time=None
):

    record = {

        "timestamp": datetime.now().isoformat(),

        "machine_id": sensor_data.get(
            "Machine_ID", "MACHINE_001"
        ),

        "machine_type": sensor_data.get("Type", "Unknown"),

        "failure_probability": triage_result.get(
            "failure_probability", 0.0
        ),

        "prediction": triage_result.get(
            "prediction", None
        ),

        "anomaly_detected": triage_result.get(
            "anomaly_detected", False
        ),

        "risk_level": triage_result.get(
            "risk_level", "UNKNOWN"
        ),

        "maintenance_priority": triage_result.get(
            "maintenance_priority", "UNKNOWN"
        ),

        "recommended_action": triage_result.get(
            "recommended_action", "None"
        ),

        "inference_time_seconds": inference_time,

        # Preserve original sensor values for later monitoring.
        "sensor_data": json.dumps(
            sensor_data,
            default=str
        )
    }

    df = pd.DataFrame([record])

    df.to_csv(
        LOG_PATH,
        mode="a",
        header=not LOG_PATH.exists(),
        index=False
    )

    return record


# --------------------------------------------------
# 3. LOAD PREDICTION LOGS
# --------------------------------------------------

def get_prediction_logs():

    if not LOG_PATH.exists():

        return pd.DataFrame()

    return pd.read_csv(LOG_PATH)


# --------------------------------------------------
# 4. MONITORING SUMMARY
# --------------------------------------------------

def get_monitoring_summary():

    df = get_prediction_logs()

    if df.empty:

        return {
            "total_predictions": 0,
            "average_failure_probability": 0,
            "anomaly_count": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0
        }

    probability = pd.to_numeric(
        df["failure_probability"],
        errors="coerce"
    )

    risk_counts = df["risk_level"].value_counts()

    anomaly_count = (
        df["anomaly_detected"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    return {

        "total_predictions": len(df),

        "average_failure_probability": round(
            probability.mean(), 4
        ),

        "anomaly_count": int(anomaly_count),

        "critical_count": int(
            risk_counts.get("CRITICAL", 0)
        ),

        "high_count": int(
            risk_counts.get("HIGH", 0)
        ),

        "medium_count": int(
            risk_counts.get("MEDIUM", 0)
        ),

        "low_count": int(
            risk_counts.get("LOW", 0)
        )
    }


# --------------------------------------------------
# 5. MODEL HEALTH CHECK
# --------------------------------------------------

def check_model_health():

    model_files = {

        "XGBoost Failure Model":
            BASE_DIR / "models" / "failure_model.pkl",

        "Feature Columns":
            BASE_DIR / "models" / "feature_columns.pkl",

        "PyTorch Autoencoder":
            BASE_DIR / "models" / "anomaly_autoencoder.pth",

        "Anomaly Scaler":
            BASE_DIR / "models" / "anomaly_scaler.pkl",

        "Anomaly Features":
            BASE_DIR / "models" / "anomaly_features.pkl",

        "Anomaly Threshold":
            BASE_DIR / "models" / "anomaly_threshold.pkl"
    }

    health_status = {}

    for model_name, model_path in model_files.items():

        health_status[model_name] = (
            "AVAILABLE"
            if model_path.exists()
            else "MISSING"
        )

    return health_status


# --------------------------------------------------
# 6. RUN MONITORING MODULE
# --------------------------------------------------

if __name__ == "__main__":

    print("\nNEUROMAINTAIN MLOPS MONITORING")

    print("=" * 40)

    print("\nMonitoring Summary:")

    summary = get_monitoring_summary()

    for key, value in summary.items():

        print(f"{key}: {value}")

    print("\nModel Health:")

    health = check_model_health()

    for name, status in health.items():

        print(f"{name}: {status}")

    print("\nRecent Prediction Logs:")

    logs = get_prediction_logs()

    if not logs.empty:

        print(logs.tail(5).to_string(index=False))

    else:

        print("No prediction records available yet.")
