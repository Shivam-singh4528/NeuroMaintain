
import pandas as pd

from src.anomaly_detection import predict_anomaly
from src.data_loader import load_data
from src.preprocessing import clean_data

# Load one sample from the dataset.
df = clean_data(load_data())

sample = df.drop(
    columns=[
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ],
    errors="ignore"
).iloc[[0]]

# Predict anomaly.
result = predict_anomaly(sample)

print("\nAnomaly Detection Result:")
print(result.to_string(index=False))
