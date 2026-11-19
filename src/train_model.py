
import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

from src.preprocessing import clean_data
from src.data_loader import load_data, BASE_DIR
from src.feature_engineering import create_features


MODEL_DIR = BASE_DIR / "models"
METRIC_DIR = BASE_DIR / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)



def prepare_data():

    df = clean_data(load_data())

    df = create_features(df)

    # Main prediction target
    y = df["Machine failure"]

    # Remove all target-related columns
    excluded_columns = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    # Encode categorical machine type
    X = pd.get_dummies(
        X,
        columns=["Type"],
        drop_first=False,
        dtype=int
    )

    # Clean feature names for XGBoost compatibility
    X.columns = (
        X.columns
        .str.replace(r"[\[\]<]", "", regex=True)
        .str.replace(" ", "_")
    )

    print("\nCleaned Feature Names:")
    print(X.columns.tolist())

    # Store feature names
    feature_columns = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns
    )



def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(
            y_test, probabilities
        )
    }

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nEvaluation Metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def train_models():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns
    ) = prepare_data()

    # Calculate class imbalance ratio
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_weight = negative_count / positive_count

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results[name] = metrics

        trained_models[name] = model

    # Select based on PR-AUC
    best_name = max(
        results,
        key=lambda name: results[name]["pr_auc"]
    )

    best_model = trained_models[best_name]

    print("\nBest model:", best_name)

    # Save model and feature information
    joblib.dump(
        best_model,
        MODEL_DIR / "failure_model.pkl"
    )

    joblib.dump(
        feature_columns,
        MODEL_DIR / "feature_columns.pkl"
    )

    joblib.dump(
        results,
        MODEL_DIR / "model_comparison.pkl"
    )

    with open(
        METRIC_DIR / "model_metrics.json",
        "w"
    ) as file:

        json.dump(results, file, indent=4)

    print("\nModels and metrics saved successfully!")


if __name__ == "__main__":

    train_models()
