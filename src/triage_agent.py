
import time

from src.predict import predict_machine
from src.anomaly_detection import detect_anomaly

from mlops.monitoring import log_prediction


# --------------------------------------------------
# 1. RISK CLASSIFICATION
# --------------------------------------------------

def classify_risk(failure_probability, anomaly_detected=False):

    if failure_probability >= 0.85:
        return "CRITICAL"

    elif failure_probability >= 0.60:
        return "HIGH"

    elif anomaly_detected or failure_probability >= 0.30:
        return "MEDIUM"

    else:
        return "LOW"


# --------------------------------------------------
# 2. MAINTENANCE RECOMMENDATIONS
# --------------------------------------------------

def get_maintenance_action(risk_level, anomaly_detected):

    if risk_level == "CRITICAL":

        return {
            "priority": "P1 - Immediate",
            "action": "Stop machine safely and arrange immediate inspection."
        }

    elif risk_level == "HIGH":

        return {
            "priority": "P2 - Urgent",
            "action": "Schedule urgent maintenance and inspect machine components."
        }

    elif anomaly_detected:

        return {
            "priority": "P2 - Inspection Required",
            "action": "Inspect unusual sensor behaviour and monitor machine closely."
        }

    elif risk_level == "MEDIUM":

        return {
            "priority": "P3 - Scheduled",
            "action": "Schedule preventive maintenance and continue monitoring."
        }

    else:

        return {
            "priority": "P4 - Routine",
            "action": "Continue normal operation and routine maintenance."
        }


# --------------------------------------------------
# 3. MAIN TRIAGE AGENT + MLOPS
# --------------------------------------------------

def run_triage(input_data):

    start_time = time.perf_counter()

    # Step A: Failure prediction
    failure_result = predict_machine(input_data)

    # Step B: Anomaly detection
    anomaly_detected = detect_anomaly(input_data)

    # Step C: Extract results
    failure_probability = float(
        failure_result["failure_probability"]
    )

    prediction = int(
        failure_result["prediction"]
    )

    # Step D: Risk classification
    risk_level = classify_risk(
        failure_probability,
        anomaly_detected
    )

    # Step E: Maintenance recommendation
    maintenance = get_maintenance_action(
        risk_level,
        anomaly_detected
    )

    # Step F: Final result
    result = {

        "prediction": prediction,

        "failure_probability": round(
            failure_probability, 4
        ),

        "risk_level": risk_level,

        "anomaly_detected": bool(anomaly_detected),

        "maintenance_priority": maintenance["priority"],

        "recommended_action": maintenance["action"]

    }

    # Step G: Inference time
    inference_time = time.perf_counter() - start_time

    # Step H: MLOps logging
    try:

        log_prediction(
            sensor_data=input_data,
            triage_result=result,
            inference_time=inference_time
        )

        result["mlops_status"] = "LOGGED"

    except Exception as error:

        print("MLOps Logging Error:", error)

        result["mlops_status"] = "LOGGING_FAILED"

    result["inference_time_seconds"] = round(
        inference_time, 4
    )

    return result


# --------------------------------------------------
# 4. TEST TRIAGE AGENT
# --------------------------------------------------

if __name__ == "__main__":

    sample_machine = {

        "Machine_ID": "MACHINE_001",

        "Type": "M",

        "Air temperature [K]": 300.5,

        "Process temperature [K]": 310.0,

        "Rotational speed [rpm]": 1500,

        "Torque [Nm]": 45,

        "Tool wear [min]": 120

    }

    print("\nNEUROMAINTAIN TRIAGE AGENT")
    print("=" * 40)

    result = run_triage(sample_machine)

    for key, value in result.items():

        print(f"{key}: {value}")
