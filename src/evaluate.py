from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "processed_churn.csv"
MODEL = ROOT / "models" / "random_forest_model.pkl"
OUT = ROOT / "outputs"


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]
    model = joblib.load(MODEL)

    # Recreate the notebook's 80/20 split.
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    (OUT / "classification_report.txt").write_text(
        f"Accuracy: {accuracy}\n\n{report}", encoding="utf-8"
    )

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()

    scores = cross_val_score(model, X, y, cv=5)
    (OUT / "cross_validation_results.txt").write_text(
        "Validation Scores: " + str(scores.tolist()) + "\n" +
        "Average Validation Accuracy: " + str(scores.mean()), encoding="utf-8"
    )

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance.sort_values(ascending=False).to_csv(OUT / "feature_importance.csv", header=["importance"])

    return accuracy, scores.mean()


if __name__ == "__main__":
    accuracy, cv_accuracy = evaluate()
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Average Validation Accuracy: {cv_accuracy:.6f}")
