
# src/anomaly_detection.py

from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data, BASE_DIR
from src.preprocessing import clean_data
from src.feature_engineering import create_features


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "anomaly_autoencoder.pth"
SCALER_PATH = MODEL_DIR / "anomaly_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "anomaly_features.pkl"
THRESHOLD_PATH = MODEL_DIR / "anomaly_threshold.pkl"

DEVICE = torch.device("cpu")

EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# --------------------------------------------------
# 2. AUTOENCODER ARCHITECTURE
# --------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, input_dim)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed


# --------------------------------------------------
# 3. FEATURE PREPARATION
# --------------------------------------------------

def prepare_features(df, feature_columns=None):

    df = df.copy()

    # Remove target and failure-mode columns.
    columns_to_drop = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "UDI",
        "Product ID"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # Create engineered sensor features.
    df = create_features(df)

    # Convert machine Type into numeric columns.
    if "Type" in df.columns:
        df = pd.get_dummies(
            df,
            columns=["Type"],
            dtype=int
        )

    # Keep numeric features only.
    df = df.select_dtypes(include=[np.number])

    # During prediction, use exactly the training columns.
    if feature_columns is not None:

        df = df.reindex(
            columns=feature_columns,
            fill_value=0
        )

    return df.astype(float)


# --------------------------------------------------
# 4. RECONSTRUCTION ERROR
# --------------------------------------------------

def calculate_reconstruction_errors(model, data):

    model.eval()

    with torch.no_grad():

        tensor_data = torch.tensor(
            data,
            dtype=torch.float32
        ).to(DEVICE)

        reconstructed = model(tensor_data)

        errors = torch.mean(
            (tensor_data - reconstructed) ** 2,
            dim=1
        )

    return errors.cpu().numpy()


# --------------------------------------------------
# 5. TRAIN AUTOENCODER
# --------------------------------------------------

def train_autoencoder():

    print("\nLoading dataset...")

    df = load_data()
    df = clean_data(df)

    if "Machine failure" not in df.columns:
        raise ValueError(
            "Machine failure column is missing from dataset."
        )

    # Train the autoencoder using normal machines only.
    normal_df = df[df["Machine failure"] == 0].copy()

    print("\nTotal records:", len(df))
    print("Normal records:", len(normal_df))

    X_normal = prepare_features(normal_df)

    feature_columns = X_normal.columns.tolist()

    # Split normal records into training and validation sets.
    X_train, X_validation = train_test_split(
        X_normal,
        test_size=0.20,
        random_state=42
    )

    # Fit scaler ONLY on training data.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    # Convert training data into PyTorch tensors.
    train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Create model.
    input_dim = X_train_scaled.shape[1]

    model = Autoencoder(input_dim).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nStarting Autoencoder training...\n")

    # Training loop.
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch in train_loader:

            inputs = batch[0].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, inputs)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Loss: {average_loss:.6f}"
            )

    # Calculate validation reconstruction errors.
    validation_errors = calculate_reconstruction_errors(
        model,
        X_validation_scaled
    )

    # Set threshold using the 99th percentile of normal validation errors.
    threshold = float(
        np.percentile(validation_errors, 99)
    )

    # Save all required artifacts.
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        feature_columns,
        FEATURES_PATH
    )

    joblib.dump(
        threshold,
        THRESHOLD_PATH
    )

    print("\nAutoencoder training completed!")

    print("Reconstruction error threshold:", threshold)

    print("\nSaved files:")
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(FEATURES_PATH)
    print(THRESHOLD_PATH)

    return {
        "threshold": threshold,
        "training_records": len(X_train),
        "validation_records": len(X_validation),
        "input_features": input_dim
    }


# --------------------------------------------------
# 6. LOAD TRAINED MODEL
# --------------------------------------------------

def load_anomaly_model():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH,
        THRESHOLD_PATH
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required model file not found: {file_path}\n"
                "Run python -m src.anomaly_detection first."
            )

    scaler = joblib.load(SCALER_PATH)

    feature_columns = joblib.load(FEATURES_PATH)

    threshold = joblib.load(THRESHOLD_PATH)

    model = Autoencoder(len(feature_columns)).to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model, scaler, feature_columns, threshold


# --------------------------------------------------
# 7. PREDICT ANOMALIES
# --------------------------------------------------

def predict_anomaly(input_data):

    # Accept either a DataFrame or a dictionary.
    if isinstance(input_data, dict):

        input_data = pd.DataFrame([input_data])

    if not isinstance(input_data, pd.DataFrame):

        raise TypeError(
            "Input must be a pandas DataFrame or dictionary."
        )

    model, scaler, feature_columns, threshold = (
        load_anomaly_model()
    )

    X = prepare_features(
        input_data,
        feature_columns=feature_columns
    )

    X_scaled = scaler.transform(X)

    errors = calculate_reconstruction_errors(
        model,
        X_scaled
    )

    results = pd.DataFrame({

        "reconstruction_error": errors,

        "is_anomaly": errors > threshold

    })

    results["status"] = np.where(
        results["is_anomaly"],
        "Anomaly Detected",
        "Normal"
    )

    return results


# --------------------------------------------------
# 8. SIMPLE ANOMALY CHECK FOR TRIAGE AGENT
# --------------------------------------------------

def detect_anomaly(input_data):

    results = predict_anomaly(input_data)

    # For one machine, return a simple True/False value.
    if len(results) == 1:

        return bool(results.iloc[0]["is_anomaly"])

    # For multiple machines, return a list.
    return results["is_anomaly"].tolist()


# --------------------------------------------------
# 9. RUN THIS FILE DIRECTLY TO TRAIN
# --------------------------------------------------

if __name__ == "__main__":

    metrics = train_autoencoder()

    print("\nTraining Summary:")
    print(metrics)
