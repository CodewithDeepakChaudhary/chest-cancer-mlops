from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/incoming/incoming_manifest.csv"
OUTPUT = ROOT / "data/artifacts/incoming_validation.json"
REQUIRED = {"image_path", "label"}
CLASSES = {"Bengin", "Malignant", "Normal"}

def main():
    if not SOURCE.exists():
        result = {
            "status": "BLOCKED",
            "reason": f"Incoming manifest not found: {SOURCE.relative_to(ROOT)}",
            "required_columns": sorted(REQUIRED),
            "expected_classes": sorted(CLASSES)
        }
        OUTPUT.write_text(json.dumps(result, indent=4))
        print(json.dumps(result, indent=4))
        return

    df = pd.read_csv(SOURCE)
    missing_columns = sorted(REQUIRED - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df["image_path"] = df["image_path"].astype(str)
    df["label"] = df["label"].astype(str)

    missing_values = int(df[["image_path", "label"]].isna().sum().sum())
    duplicate_paths = int(df["image_path"].duplicated().sum())
    invalid_labels = sorted(set(df["label"]) - CLASSES)

    def resolve_path(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    missing_images = [
        value for value in df["image_path"]
        if not resolve_path(value).is_file()
    ]

    passed = (
        len(df) > 0 and missing_values == 0 and
        duplicate_paths == 0 and not invalid_labels and
        not missing_images
    )

    result = {
        "status": "PASS" if passed else "FAIL",
        "rows": len(df),
        "missing_values": missing_values,
        "duplicate_image_paths": duplicate_paths,
        "invalid_labels": invalid_labels,
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:10],
        "class_distribution": df["label"].value_counts().to_dict()
    }

    OUTPUT.write_text(json.dumps(result, indent=4))
    print(json.dumps(result, indent=4))

    if not passed:
        raise SystemExit("Incoming data validation failed.")

if __name__ == "__main__":
    main()
