from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data/artifacts"
CHAMPION = ARTIFACTS / "champion_evaluation.json"
CANDIDATE = ARTIFACTS / "candidate_evaluation.json"
TRAINING_STATUS = ARTIFACTS / "candidate_training_status.json"
OUTPUT = ARTIFACTS / "quality_gate_decision.json"

def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

def main():
    champion = read_json(CHAMPION)
    candidate = read_json(CANDIDATE)
    training = read_json(TRAINING_STATUS)

    decision = {
        "champion_model": "models/chest_cancer_model.keras",
        "candidate_model": "models/candidates/candidate_model.keras",
        "promotion_metric": "Macro F1",
        "criteria": {
            "candidate_macro_f1_must_be_at_least_champion": True,
            "candidate_malignant_recall_must_not_decrease": True
        },
        "promotion_status": "HOLD",
        "promotion_eligible": False
    }

    if not champion:
        decision["reason"] = "Champion evaluation report is missing."
    elif not candidate or (training and training.get("status") != "SUCCESS"):
        decision["reason"] = (
            "Candidate training/evaluation is unavailable. "
            "Champion remains unchanged."
        )
    else:
        champion_f1 = champion.get("macro_f1", 0)
        candidate_f1 = candidate.get("macro_f1", 0)

        champion_recall = champion.get("per_class", {}).get(
            "Malignant", {}
        ).get("recall")
        candidate_recall = candidate.get("per_class", {}).get(
            "Malignant", {}
        ).get("recall")

        # Support sklearn's classification_report structure.
        if champion_recall is None:
            champion_recall = champion.get("per_class", {}).get(
                "Malignant", {}
            ).get("recall", 0)
        if candidate_recall is None:
            candidate_recall = candidate.get("per_class", {}).get(
                "Malignant", {}
            ).get("recall", 0)

        passed = (
            candidate_f1 >= champion_f1 and
            candidate_recall is not None and
            champion_recall is not None and
            candidate_recall >= champion_recall
        )

        decision.update({
            "champion_macro_f1": champion_f1,
            "candidate_macro_f1": candidate_f1,
            "champion_malignant_recall": champion_recall,
            "candidate_malignant_recall": candidate_recall,
            "promotion_eligible": bool(passed),
            "promotion_status": "ELIGIBLE_FOR_REVIEW" if passed else "HOLD",
            "reason": (
                "Candidate meets both quality criteria; manual approval is still required."
                if passed else
                "Candidate failed at least one quality criterion. Champion retained."
            )
        })

    OUTPUT.write_text(json.dumps(decision, indent=4))
    print(json.dumps(decision, indent=4))
    print(f"\nSaved: {OUTPUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
