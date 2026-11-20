from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/artifacts/test_predictions.csv"
OUTPUT = ROOT / "data/artifacts/champion_evaluation.json"

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}")

    df = pd.read_csv(SOURCE)
    required = {"actual_label", "predicted_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")

    df = df.dropna(subset=["actual_label", "predicted_label"])
    y_true = df["actual_label"].astype(str)
    y_pred = df["predicted_label"].astype(str)

    labels = ["Bengin", "Malignant", "Normal"]
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True,
        zero_division=0
    )

    result = {
        "model": "models/chest_cancer_model.keras",
        "evaluation_source": "existing test_predictions.csv",
        "note": "This evaluates saved predictions; it is not a fresh inference run.",
        "samples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "per_class": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist()
    }

    OUTPUT.write_text(json.dumps(result, indent=4))
    print(json.dumps(result, indent=4))
    print(f"\nSaved: {OUTPUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
