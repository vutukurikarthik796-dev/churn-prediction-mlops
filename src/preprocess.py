import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "churn.csv"
PROCESSED = ROOT / "data" / "processed" / "processed_churn.csv"
METADATA = ROOT / "data" / "processed" / "dataset_metadata.json"
ARTIFACT = ROOT / "artifacts" / "preprocessing_metadata.json"


def preprocess():
    df = pd.read_csv(RAW)
    original_shape = list(df.shape)

    # Same preprocessing steps used in the notebook.
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_before_fill = int(df["TotalCharges"].isna().sum())
    median_total_charges = float(df["TotalCharges"].median())
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)
    df = pd.get_dummies(df, drop_first=True)
    df["AverageCharges"] = df["TotalCharges"] / (df["tenure"] + 1)

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)

    metadata = {
        "original_shape": original_shape,
        "processed_shape": list(df.shape),
        "dropped_columns": ["customerID"],
        "numeric_conversion": {"TotalCharges": "pd.to_numeric(errors='coerce')"},
        "missing_totalcharges_before_fill": missing_before_fill,
        "totalcharges_fill": "median",
        "one_hot_encoding": "pd.get_dummies(drop_first=True)",
        "engineered_features": ["AverageCharges"],
        "target": "Churn_Yes",
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ARTIFACT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    df = preprocess()
    print(f"Processed dataset saved to {PROCESSED}")
    print(f"Shape: {df.shape}")
