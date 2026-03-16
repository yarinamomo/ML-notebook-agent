import argparse
import json
from pathlib import Path
from typing import Any


def is_filled_label(label: Any) -> bool:
    """Return True if a manual label is considered filled."""
    if label is None:
        return False
    if isinstance(label, str):
        return label.strip() != ""
    return True


def compute_manual_validation(sampled_instances: dict[str, Any]) -> dict[str, Any]:
    """Compute manual validation status and valid percentage for sampled instances."""
    labels = list(sampled_instances.values())
    status = "done" if all(is_filled_label(label) for label in labels) else "incomplete"

    if status != "done":
        return {"status": status, "validation_outcome": None}

    total = len(labels)
    valid_count = sum(
        1 for label in labels if isinstance(label, str) and label.strip().lower() == "valid"
    )
    validation_outcome = f"{valid_count}/{total} ({round((valid_count / total * 100.0), 2)}%)"
    return {"status": status, "validation_outcome": validation_outcome}


def update_manual_validation_in_file(
    setting: str,
    model: str,
    results_root: str | Path = "results",
) -> Path:
    """Update manual validation metrics in the labeled sampled-instances JSON file."""
    json_path = Path(results_root) / setting / model / "analysis" / "stratified_sampled_instances_labeled.json"
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    sampled_instances = data.get("sampled_instances")
    if not isinstance(sampled_instances, dict):
        raise ValueError(f"sampled_instances must be a dictionary in {json_path}")

    manual_validation = compute_manual_validation(sampled_instances)

    updated_data = {"manual_validation": manual_validation}
    for key, value in data.items():
        if key != "manual_validation":
            updated_data[key] = value

    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(updated_data, file_obj, indent=2)
        file_obj.write("\n")

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate and store manual validation percentage for sampled instances.",
    )
    parser.add_argument("--setting", required=True, help="Result setting folder under results/.")
    parser.add_argument(
        "--model",
        default="glm-4.7-355b",
        help="Model folder under results/<setting>/ (default: glm-4.7-355b).",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing result folders (default: results).",
    )
    args = parser.parse_args()

    updated_path = update_manual_validation_in_file(
        setting=args.setting,
        model=args.model,
        results_root=args.results_root,
    )
    print(f"Updated manual validation metrics in: {updated_path}")


if __name__ == "__main__":
    main()
