import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "processed_churn.csv"
MODEL = ROOT / "models" / "random_forest_model.pkl"
SPLIT = ROOT / "artifacts" / "split_data.json"


def train():
    df = pd.read_csv(DATA)
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    SPLIT.write_text(json.dumps({
        "test_size": 0.2,
        "random_state": 42,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape)
    }, indent=2), encoding="utf-8")
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train()
    print(f"Model saved to {MODEL}")
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
