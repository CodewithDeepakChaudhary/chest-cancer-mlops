# Chest Cancer MLOps – Model Monitoring and Retraining Assessment

## 1. Objective

Extend the existing chest cancer classification project with incoming-data
validation, champion evaluation, candidate retraining, and a model quality gate.

## 2. Existing Model

- Architecture: MobileNetV2
- Input image size: 160 x 160
- Classes: Bengin, Malignant, Normal
- Champion artifact: models/chest_cancer_model.keras

## 3. Task A – Incoming Data Validation

Script: scripts/prepare_incoming_data.py

The validation workflow checks:
- Required columns: image_path and label
- Missing values
- Duplicate image paths
- Valid class labels
- Image file availability
- Class distribution

Current result: BLOCKED

Reason: data/incoming/incoming_manifest.csv is not available.
The validation result is saved in:
data/artifacts/incoming_validation.json

## 4. Task B – Champion Model Evaluation

Script: scripts/evaluate_champion.py

The existing saved predictions are evaluated using:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Per-class classification metrics
- Confusion matrix

Evaluation input:
data/artifacts/test_predictions.csv

Evaluation report:
data/artifacts/champion_evaluation.json

Important: This evaluates previously saved predictions. It is not a
fresh inference run on a new incoming dataset.

## 5. Task C – Candidate Model Retraining

Script: scripts/retrain_candidate.py

The candidate training workflow:
- Loads existing train, validation, and test datasets.
- Combines training data with the incoming labelled batch.
- Removes duplicate image paths.
- Uses MobileNetV2 preprocessing and architecture.
- Applies class weights.
- Uses early stopping and learning-rate reduction.
- Saves the candidate separately from the champion.
- Evaluates the candidate on the held-out test dataset.

Current result: BLOCKED

Reasons:
- data/processed/train.csv is missing.
- data/processed/val.csv is missing.
- data/processed/test.csv is missing.
- data/incoming/incoming_manifest.csv is missing.
- TensorFlow is not installed in the current environment.

Candidate artifact has not been generated.

## 6. Task D – Model Quality Gate

Script: scripts/quality_gate.py

Promotion criteria:
1. Candidate Macro F1 must be greater than or equal to champion Macro F1.
2. Candidate Malignant recall must be greater than or equal to champion
   Malignant recall.

Current decision: HOLD

Promotion eligible: false

Reason: Candidate training and evaluation are unavailable.

The existing champion model remains unchanged. The quality gate does not
automatically replace the production model.

Decision report:
data/artifacts/quality_gate_decision.json

## 7. Task E – Execution Commands

Run the following commands from the project root:

    python scripts/prepare_incoming_data.py
    python scripts/evaluate_champion.py
    python scripts/retrain_candidate.py
    python scripts/quality_gate.py

## 8. Generated Evidence

- data/artifacts/incoming_validation.json
- data/artifacts/champion_evaluation.json
- data/artifacts/candidate_training_status.json
- data/artifacts/quality_gate_decision.json

## 9. Conclusion

The monitoring and quality-gate workflow has been implemented and executed.
The existing champion evaluation report and HOLD decision were generated.

A successful retraining and promotion experiment remains pending because
the original image dataset, processed datasets, and incoming labelled batch
are unavailable in the current Codespace.

The champion has been retained safely. No fabricated candidate metrics or
automatic model promotion were used.
