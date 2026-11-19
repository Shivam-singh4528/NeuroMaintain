
import numpy as np
import pandas as pd


def create_features(df):

    df = df.copy()

    # Temperature difference
    df["Temperature Difference"] = (
        df["Process temperature [K]"]
        - df["Air temperature [K]"]
    )

    # Mechanical power estimate in watts
    df["Power Estimate [W]"] = (
        df["Torque [Nm]"]
        * df["Rotational speed [rpm]"]
        * (2 * np.pi / 60)
    )

    # Interaction feature
    df["Torque RPM Interaction"] = (
        df["Torque [Nm]"]
        * df["Rotational speed [rpm]"]
    )

    # Temperature ratio
    df["Temperature Ratio"] = (
        df["Process temperature [K]"]
        / df["Air temperature [K]"]
    )

    print("\nFeature engineering completed!")

    print("\nNew features:")
    print([
        "Temperature Difference",
        "Power Estimate [W]",
        "Torque RPM Interaction",
        "Temperature Ratio"
    ])

    return df
