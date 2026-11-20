# MLOps Practical Assessment

## Project
Chest Cancer Classification using MobileNetV2, TensorFlow and Keras.

## Task A — Prepare New Data
Validated 159 existing labelled prediction records.
- Missing values: 0
- Duplicate records: 0
- Invalid labels: 0
- Malignant: 82
- Normal: 61
- Bengin: 16

Note: This is a replay of existing test predictions, not a new independent incoming batch.

## Task B — Evaluate Existing Model
- Accuracy: 94.97%
- Macro Precision: 89.60%
- Macro Recall: 89.48%
- Macro F1: 89.54%

## Task C — Retrain Model
Candidate training is BLOCKED because original image and training data are unavailable. No candidate model was fabricated.

## Task D — Quality Gate
- Primary metric: Macro F1
- Candidate status: BLOCKED
- Promotion status: HOLD
- Promotion eligible: false
- Champion retained: models/chest_cancer_model.keras

Macro F1 is selected because it gives equal importance to all three classes.

## Task E — Demonstration
Run: python scripts/model_monitoring.py

Inspect:
- data/artifacts/monitoring_validation.json
- data/artifacts/incoming_batch_metrics.json
- data/artifacts/quality_gate_decision.json

## Remaining work
A fresh incoming image batch and candidate retraining are required to complete Tasks B and C fully.
