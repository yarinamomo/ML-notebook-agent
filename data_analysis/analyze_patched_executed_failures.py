from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from enum import Enum

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.util_test_cells_evaluation import VALIDATED_TEST_NOTEBOOK_EXCLUSIONS


DEFAULT_ROOT = Path("results")
DEFAULT_OUTPUT_PATH = Path("results/data_analysis/patched_executed_failures_summary.json")

class Classification(Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    TRUE_NEGATIVE = "true_negative"


@dataclass(slots=True)
class NotebookFailureRecord:
    setting: str
    model: str
    run: str
    notebook_path: str
    last_cell_was_executed: bool
    last_cell_has_error: bool
    error_name: str | None
    error_preview: str | None
    summary_success: bool | None
    classification: Classification | None


def _is_test_failure(record: NotebookFailureRecord) -> bool:
    return record.last_cell_has_error or not record.last_cell_was_executed


def _classify_record(predicted_success: bool, actual_success: bool) -> Classification:
    if predicted_success and actual_success:
        return Classification.TRUE_POSITIVE
    if predicted_success and not actual_success:
        return Classification.FALSE_POSITIVE
    if not predicted_success and actual_success:
        return Classification.FALSE_NEGATIVE
    return Classification.TRUE_NEGATIVE


def _confusion_counts(records: list[NotebookFailureRecord]) -> dict[str, float | int]:
    counter = Counter(record.classification for record in records if record.classification is not None)
    tp = counter[Classification.TRUE_POSITIVE]
    fp = counter[Classification.FALSE_POSITIVE]
    fn = counter[Classification.FALSE_NEGATIVE]
    tn = counter[Classification.TRUE_NEGATIVE]
    labeled_instances = tp + fp + fn + tn

    return {
        "labeled_instances": labeled_instances,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else -1,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else -1,
    }


def _record_to_dict(record: NotebookFailureRecord) -> dict[str, Any]:
    return {
        "setting": record.setting,
        "model": record.model,
        "run": record.run,
        "notebook_path": record.notebook_path,
        "last_cell_was_executed": record.last_cell_was_executed,
        "last_cell_has_error": record.last_cell_has_error,
        "error_name": record.error_name,
        "error_preview": record.error_preview,
        "summary_success": record.summary_success,
        "classification": record.classification.value if record.classification is not None else None,
    }

def _instance_name_from_patched_executed_path(notebook_path: Path) -> str:
    stem = notebook_path.stem
    suffix = "_patched_executed"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _discover_notebooks() -> list[Path]:
    discovered: list[Path] = []
    excluded_instances = {name.lower() for name in VALIDATED_TEST_NOTEBOOK_EXCLUSIONS}
    root = DEFAULT_ROOT
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"

    for notebook_path in sorted(root.glob("*/*/run_*/*_patched_executed.ipynb")):
        instance_name = _instance_name_from_patched_executed_path(notebook_path)
        if instance_name.lower() in excluded_instances:
            continue
        discovered.append(notebook_path)

    return discovered


def _summary_path_from_notebook_path(notebook_path: Path) -> Path:
    stem = notebook_path.stem
    suffix = "_patched_executed"
    if stem.endswith(suffix):
        instance_name = stem[: -len(suffix)]
    else:
        instance_name = stem
    return notebook_path.with_name(f"{instance_name}_summary.json")


def _read_summary_success(notebook_path: Path) -> bool:
    summary_path = _summary_path_from_notebook_path(notebook_path)
    if not summary_path.exists():
        raise ValueError(f"Summary file not found: {summary_path}")

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Error reading summary file: {summary_path}") from e

    metadata = payload.get("metadata", {})
    success = metadata.get("success")
    if isinstance(success, bool):
        return success
    raise ValueError(f"Invalid summary success value: {success}")


def _parse_notebook_identity(notebook_path: Path) -> tuple[str, str, str]:
    relative = notebook_path.relative_to(DEFAULT_ROOT)
    parts = relative.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected notebook path layout: {notebook_path}")

    setting, model, run = parts[0], parts[1], parts[2]
    return setting, model, run


def _check_last_cell_status(cell: dict[str, Any] | None) -> tuple[bool, bool, str | None, str | None]:
    """Check if last cell was executed and if it has an error.
    
    Returns: (was_executed, has_error, error_type, error_preview)
    """
    if cell is None:
        return False, False, None, None

    # Check if cell was executed
    was_executed = cell.get("execution_count") is not None

    # Check for errors in outputs
    outputs = cell.get("outputs", [])
    for output in outputs:
        if output.get("output_type") != "error":
            continue

        ename = str(output.get("ename", "")).strip() or None
        evalue = str(output.get("evalue", "")).strip()
        traceback_lines = [str(line) for line in output.get("traceback", [])]

        preview = evalue if evalue else None
        if not preview and traceback_lines:
            preview = traceback_lines[-1]

        return was_executed, True, ename, preview

    return was_executed, False, None, None


def _safe_pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _to_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def _round(value: float) -> float:
    return round(value, 4)


def analyze_notebooks() -> dict[str, Any]:
    records: list[NotebookFailureRecord] = []
    notebooks = _discover_notebooks()

    for notebook_path in notebooks:
        setting, model, run = _parse_notebook_identity(notebook_path)
        notebook = nbformat.read(notebook_path, as_version=4)
        last_cell = notebook.cells[-1] if notebook.cells else None
        assert last_cell is not None, f"No cells found in notebook: {notebook_path}"
        assert last_cell.get("cell_type") == "code", f"Last cell is not a code cell in notebook: {notebook_path}"
        was_executed, has_error, error_name, error_preview = _check_last_cell_status(last_cell)
        summary_success = _read_summary_success(notebook_path)
        actual_success = not has_error and was_executed

        classification = _classify_record(predicted_success=summary_success, actual_success=actual_success)

        records.append(
            NotebookFailureRecord(
                setting=setting,
                model=model,
                run=run,
                notebook_path=str(notebook_path.as_posix()),
                last_cell_was_executed=was_executed,
                last_cell_has_error=has_error,
                error_name=error_name,
                error_preview=error_preview,
                summary_success=summary_success,
                classification=classification,
            )
        )

    total = len(records)
    total_with_test_failure = sum(1 for record in records if _is_test_failure(record))
    total_test_failure_success_true = sum(
        1
        for record in records
        if _is_test_failure(record) and record.summary_success is True
    )

    by_group: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[NotebookFailureRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.setting, record.model)].append(record)

    for (setting, model), group_records in sorted(grouped.items()):
        run_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "failing": 0})
        error_types: Counter[str] = Counter()

        for record in group_records:
            run_counts[record.run]["total"] += 1
            if _is_test_failure(record):
                run_counts[record.run]["failing"] += 1
                error_key = record.error_name or ("NotExecuted" if not record.last_cell_was_executed else "UnknownError")
                error_types[error_key] += 1

        total_group = len(group_records)
        failing_group = sum(1 for record in group_records if _is_test_failure(record))
        test_failure_group_success_true = sum(
            1
            for record in group_records
            if _is_test_failure(record) and record.summary_success is True
        )

        run_failure_rates = []
        run_details: dict[str, Any] = {}
        for run_name in sorted(run_counts):
            run_total = run_counts[run_name]["total"]
            run_failing = run_counts[run_name]["failing"]
            fail_rate = run_failing / run_total if run_total else 0.0
            run_failure_rates.append(fail_rate)
            run_details[run_name] = {
                "total": run_total,
                "failing": run_failing,
                "failing_rate": _round(fail_rate),
            }

        key = f"{setting}/{model}"
        by_group[key] = {
            "setting": setting,
            "model": model,
            "total": total_group,
            "failing": failing_group,
            "failing_percent": _round(_to_percent(failing_group, total_group)),
            "test_failures_success_true": test_failure_group_success_true,
            "test_failure_percent_success_true": _round(_to_percent(test_failure_group_success_true, failing_group)),
            "error_types": dict(error_types),
            "confusion_matrix": _confusion_counts(group_records),
            "run_to_run_variance": {
                "n_runs": len(run_failure_rates),
                "mean_failure_rate": _round(_safe_mean(run_failure_rates)),
                "stdev_failure_rate": _round(_safe_pstdev(run_failure_rates)),
                "variance_failure_rate": _round(_safe_pstdev(run_failure_rates) ** 2),
                "min_failure_rate": _round(min(run_failure_rates) if run_failure_rates else 0.0),
                "max_failure_rate": _round(max(run_failure_rates) if run_failure_rates else 0.0),
            },
            "per_run": run_details,
        }

    non_assertion_examples = [
        {
            "path": record.notebook_path,
            "error_name": record.error_name,
            "error_preview": record.error_preview,
        }
        for record in records
        if _is_test_failure(record)
    ]

    return {
        "summary": {
            "excluded_instances": sorted(VALIDATED_TEST_NOTEBOOK_EXCLUSIONS),
            "notebooks_analyzed": total,
            "test_failure_count": total_with_test_failure,
            "test_failure_percent": _round(_to_percent(total_with_test_failure, total)),
            "test_failures_success_true": total_test_failure_success_true,
            "confusion_matrix": _confusion_counts(records),
        },
        "by_group": by_group,
        "failure_examples": non_assertion_examples,
        "records": [_record_to_dict(record) for record in records],
    }


def _print_console_summary(analysis: dict[str, Any]) -> None:
    summary = analysis["summary"]
    confusion = summary["confusion_matrix"]
    print("=" * 88)
    print("Patched Executed Notebook Failure Analysis")
    print("=" * 88)
    print(f"Notebooks analyzed: {summary['notebooks_analyzed']}")
    print(
        "Test failures (last cell error or not executed): "
        f"{summary['test_failure_count']} ({summary['test_failure_percent']}%)"
    )
    print(
        "Test failures among success=true notebooks: "
        f"{summary['test_failures_success_true']}"
    )
    print(
        "FP/FN (prediction=test failure, actual=not success): "
        f"FP={confusion['false_positive']} FN={confusion['false_negative']} "
        f"FPR={confusion['false_positive_rate']}% FNR={confusion['false_negative_rate']}%"
    )
    print("-" * 88)
    print("Per type/model")
    print("-" * 88)

    for key, group in analysis["by_group"].items():
        variance = group["run_to_run_variance"]
        cm = group["confusion_matrix"]
        print(
            f"{key}: total={group['total']} failing={group['failing']} "
            f"({group['failing_percent']}%), test_fail_success_true={group['test_failures_success_true']}, "
            f"runs={variance['n_runs']} mean_fail_rate={variance['mean_failure_rate']}, "
            f"stdev={variance['stdev_failure_rate']}, min={variance['min_failure_rate']}, "
            f"max={variance['max_failure_rate']}, FP={cm['false_positive']} FN={cm['false_negative']}"
        )

    if summary["test_failure_count"] > 0:
        print("-" * 88)
        print("Error types by group:")
        for key, group in analysis["by_group"].items():
            if not group["error_types"]:
                continue
            print(f"{key}: {group['error_types']}")


def main() -> None:
    analysis = analyze_notebooks()

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    _print_console_summary(analysis)
    print("-" * 88)
    print(f"Wrote JSON summary to: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
