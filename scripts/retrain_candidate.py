from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data/artifacts"
OUTPUT = ARTIFACTS / "candidate_training_status.json"

def save_blocked(reason):
    result = {"status": "BLOCKED", "reason": reason}
    OUTPUT.write_text(json.dumps(result, indent=4))
    print(json.dumps(result, indent=4))
    raise SystemExit(1)

def main():
    train_file = ROOT / "data/processed/train.csv"
    val_file = ROOT / "data/processed/val.csv"
    test_file = ROOT / "data/processed/test.csv"
    incoming_file = ROOT / "data/incoming/incoming_manifest.csv"

    missing = [
        str(p.relative_to(ROOT))
        for p in [train_file, val_file, test_file, incoming_file]
        if not p.exists()
    ]
    if missing:
        save_blocked("Required datasets are missing: " + ", ".join(missing))

    try:
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
    except ImportError as exc:
        save_blocked(f"Training dependency unavailable: {exc}")

    try:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        incoming_df = pd.read_csv(incoming_file)

        required = {"image_path", "label"}
        for name, frame in [
            ("train", train_df), ("validation", val_df),
            ("test", test_df), ("incoming", incoming_df)
        ]:
            if not required.issubset(frame.columns):
                raise ValueError(f"{name}.csv must contain {sorted(required)}")

        combined = pd.concat([train_df, incoming_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["image_path"])
        classes = ["Bengin", "Malignant", "Normal"]

        if set(combined["label"].astype(str)) - set(classes):
            raise ValueError("Unexpected class labels in training data.")

        def resolve(p):
            path = Path(str(p))
            return path if path.is_absolute() else ROOT / path

        for frame_name, frame in [
            ("combined training", combined), ("validation", val_df),
            ("test", test_df)
        ]:
            missing_paths = [
                str(p) for p in frame["image_path"] if not resolve(p).is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(
                    f"{frame_name}: {len(missing_paths)} image files missing; "
                    f"examples: {missing_paths[:5]}"
                )

        # Keep the existing validation and test splits separate.
        def load_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, (160, 160))
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return image, label

        def make_dataset(frame, shuffle=False):
            label_map = {name: i for i, name in enumerate(classes)}
            paths = [str(resolve(p)) for p in frame["image_path"]]
            labels = [label_map[str(v)] for v in frame["label"]]
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            if shuffle:
                ds = ds.shuffle(len(paths), seed=42)
            return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE) \
                     .batch(8).prefetch(tf.data.AUTOTUNE)

        train_ds = make_dataset(combined, True)
        val_ds = make_dataset(val_df)
        test_ds = make_dataset(test_df)

        y_train = combined["label"].map(
            {name: i for i, name in enumerate(classes)}
        ).to_numpy()
        weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(3), y=y_train
        )
        class_weights = {i: float(w) for i, w in enumerate(weights)}

        base = tf.keras.applications.MobileNetV2(
            input_shape=(160, 160, 3), include_top=False,
            weights="imagenet"
        )
        base.trainable = False

        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        candidate_dir = ROOT / "models/candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / "candidate_model.keras"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(candidate_path), monitor="val_loss",
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=2, factor=0.5
            )
        ]

        history = model.fit(
            train_ds, validation_data=val_ds, epochs=20,
            class_weight=class_weights, callbacks=callbacks
        )

        model = tf.keras.models.load_model(candidate_path)
        probabilities = model.predict(test_ds)
        predictions = probabilities.argmax(axis=1)
        actual = np.concatenate([y.numpy() for _, y in test_ds])
        labels = np.arange(3)

        result = {
            "model": str(candidate_path.relative_to(ROOT)),
            "samples": int(len(actual)),
            "accuracy": float(accuracy_score(actual, predictions)),
            "macro_precision": float(precision_score(
                actual, predictions, labels=labels,
                average="macro", zero_division=0
            )),
            "macro_recall": float(recall_score(
                actual, predictions, labels=labels,
                average="macro", zero_division=0
            )),
            "macro_f1": float(f1_score(
                actual, predictions, labels=labels,
                average="macro", zero_division=0
            )),
            "per_class": classification_report(
                actual, predictions, labels=labels,
                target_names=classes, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                actual, predictions, labels=labels
            ).tolist(),
            "training_epochs": len(history.history["loss"])
        }

        (ARTIFACTS / "candidate_evaluation.json").write_text(
            json.dumps(result, indent=4)
        )
        OUTPUT.write_text(json.dumps({
            "status": "SUCCESS",
            "candidate": result["model"]
        }, indent=4))
        print(json.dumps(result, indent=4))
        print("\nCandidate training completed.")

    except Exception as exc:
        save_blocked(f"{type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
