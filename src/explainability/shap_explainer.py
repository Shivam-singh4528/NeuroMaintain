
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.data_loader import BASE_DIR
from src.feature_engineering import create_features


# Model paths
MODEL_PATH = BASE_DIR / "models" / "failure_model.pkl"

FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

OUTPUT_PATH = BASE_DIR / "shap_explanation.png"


def explain_prediction(sensor_data):

    # 1. Load trained model
    model = joblib.load(MODEL_PATH)

    # 2. Load training feature columns
    feature_columns = joblib.load(FEATURE_PATH)

    # 3. Convert sensor input into DataFrame
    df = pd.DataFrame([sensor_data])

    # 4. Apply same feature engineering as predict.py
    df = create_features(df)

    # 5. Encode machine Type
    df = pd.get_dummies(
        df,
        columns=["Type"],
        dtype=int
    )

    # 6. Match training columns
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # 7. Create SHAP Tree Explainer
    explainer = shap.TreeExplainer(model)

    # 8. Calculate SHAP values
    shap_values = explainer.shap_values(df)

    # 9. Select failure class (class 1)
    classes = list(model.classes_)
    class_index = classes.index(1)

    if isinstance(shap_values, list):

        values = np.asarray(
            shap_values[class_index]
        )[0]

    else:

        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, class_index]
        else:
            values = values[0]

    # 10. Get base value
    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):

        expected_value = np.asarray(expected_value).reshape(-1)

        if len(expected_value) > 1:
            base_value = float(expected_value[class_index])
        else:
            base_value = float(expected_value[0])

    else:
        base_value = float(expected_value)

    # 11. Create SHAP Explanation object
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=df.iloc[0].values,
        feature_names=list(df.columns)
    )

    # 12. Generate Waterfall Plot
    plt.figure(figsize=(12, 7))

    shap.plots.waterfall(
        explanation,
        max_display=10,
        show=False
    )

    plt.title(
        "NeuroMaintain - Failure Prediction Explanation"
    )

    plt.tight_layout()

    # 13. Save explanation plot
    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 14. Get top contributing features
    importance = pd.DataFrame({

        "Feature": df.columns,

        "SHAP_Value": values,

        "Impact": np.abs(values)

    })

    importance = importance.sort_values(
        by="Impact",
        ascending=False
    )

    return {
        "explanation": explanation,
        "top_features": importance.head(10),
        "plot_path": str(OUTPUT_PATH)
    }


if __name__ == "__main__":

    sample_machine = {

        "Type": "M",

        "Air temperature [K]": 300.5,

        "Process temperature [K]": 310.0,

        "Rotational speed [rpm]": 1500,

        "Torque [Nm]": 45,

        "Tool wear [min]": 120

    }

    result = explain_prediction(sample_machine)

    print("\nSHAP EXPLANATION GENERATED")

    print("\nTop Important Features:")

    print(result["top_features"].to_string(index=False))

    print("\nPlot saved at:")

    print(result["plot_path"])
