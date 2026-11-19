
import joblib
import pandas as pd

from src.data_loader import BASE_DIR
from src.feature_engineering import create_features


MODEL_PATH = BASE_DIR / "models" / "failure_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

THRESHOLD = 0.5


def clean_feature_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.replace(r"[\[\]<>*]", "", regex=True)
        .str.replace(" ", "_", regex=False)
    )

    return df


def predict_machine(sensor_data):

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Load training feature names
    feature_columns = joblib.load(FEATURE_PATH)

    # Convert input into DataFrame
    df = pd.DataFrame([sensor_data])

    # Generate engineered features
    df = create_features(df)

    # Encode categorical machine type
    df = pd.get_dummies(
        df,
        columns=["Type"],
        dtype=int
    )

    # IMPORTANT: Apply same feature-name cleaning as training
    df = clean_feature_names(df)

    # Check missing features before alignment
    missing_features = [
        col for col in feature_columns
        if col not in df.columns
    ]

    extra_features = [
        col for col in df.columns
        if col not in feature_columns
    ]

    print("\nMissing Features:", missing_features)
    print("Extra Features:", extra_features)

    # Match exact training columns and order
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict failure probability
    probability = model.predict_proba(df)[0][1]

    # Classification
    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "failure_probability": float(probability),
        "threshold": THRESHOLD
    }


if __name__ == "__main__":

    sample_machine = {
    "Type": "L",
    "Air temperature [K]": 303.0,
    "Process temperature [K]": 313.0,
    "Rotational speed [rpm]": 1400,
    "Torque [Nm]": 55,
    "Tool wear [min]": 240
}
    result = predict_machine(sample_machine)

    print("\n========== NeuroMaintain Prediction ==========")

    print("Prediction:", result["prediction"])

    print(
        "Failure Probability:",
        round(result["failure_probability"] * 100, 2),
        "%"
    )

    if result["prediction"] == 1:
        print("Machine Failure Risk Detected!")
    else:
        print("Machine Operating Normally!")

    print("==============================================")
