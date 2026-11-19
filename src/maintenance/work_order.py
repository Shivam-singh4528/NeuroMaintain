
# src/maintenance/work_order.py

import json
import uuid

from datetime import datetime
from pathlib import Path

from src.data_loader import BASE_DIR


WORK_ORDER_DIR = BASE_DIR / "mlops" / "work_orders"
WORK_ORDER_DIR.mkdir(parents=True, exist_ok=True)


def generate_work_order(triage_result, machine_data):

    risk = triage_result["risk_level"]

    if risk == "CRITICAL":
        priority = "P1 - Immediate"
        instructions = [
            "Safely stop the machine",
            "Arrange immediate inspection",
            "Inspect critical machine components",
            "Verify machine safety before restarting"
        ]

    elif risk == "HIGH":
        priority = "P2 - Urgent"
        instructions = [
            "Schedule urgent maintenance",
            "Inspect machine components",
            "Check abnormal sensor readings"
        ]

    elif risk == "MEDIUM":
        priority = "P3 - Scheduled"
        instructions = [
            "Schedule preventive maintenance",
            "Inspect machine condition",
            "Continue sensor monitoring"
        ]

    else:
        priority = "P4 - Routine"
        instructions = [
            "Continue normal operation",
            "Perform routine inspection",
            "Monitor sensor readings"
        ]

    if triage_result["anomaly_detected"]:
        instructions.append(
            "Investigate unusual sensor behaviour"
        )

    work_order = {

        "work_order_id": str(uuid.uuid4())[:8],

        "machine_id": machine_data.get(
            "Machine_ID", "MACHINE_001"
        ),

        "created_at": datetime.now().isoformat(),

        "risk_level": risk,

        "priority": priority,

        "failure_probability": triage_result[
            "failure_probability"
        ],

        "anomaly_detected": triage_result[
            "anomaly_detected"
        ],

        "recommended_action": triage_result[
            "recommended_action"
        ],

        "instructions": instructions,

        "status": "PENDING"
    }

    file_path = (
        WORK_ORDER_DIR /
        f"{work_order['work_order_id']}.json"
    )

    with open(file_path, "w") as file:
        json.dump(work_order, file, indent=4)

    return work_order


if __name__ == "__main__":

    sample_triage = {
        "risk_level": "HIGH",
        "failure_probability": 0.75,
        "anomaly_detected": True,
        "recommended_action": "Urgent maintenance required"
    }

    sample_machine = {
        "Machine_ID": "MACHINE_001"
    }

    print(generate_work_order(
        sample_triage,
        sample_machine
    ))
