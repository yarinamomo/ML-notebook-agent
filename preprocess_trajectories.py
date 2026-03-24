#!/usr/bin/env python3
"""
Preprocess trajectory data into a single JSON file for the viewer app.
Reads summary JSON files from the trajectories_remote/ directory
and aggregates them into viewer/public/data.json.
"""

import json
import os
import sys
from pathlib import Path

from src.utils.format_nb_cells import format_cell_source_for_llm
from src.utils.nbformat_helper import load_and_parse_notebook


REFERENCE_NOTEBOOK_CACHE: dict[str, list[str] | None] = {}


def classify_action(action: str) -> str:
    """Classify the action type from an operation's action string."""
    if "edit_cell" in action:
        return "edit_cell"
    elif "run_all" in action:
        return "run_all"
    elif "run_cell" in action:
        return "run_cell"
    elif "get_cell(" in action:
        return "get_cell"
    elif "get_cells" in action:
        return "get_cells"
    elif "get_cell_count" in action:
        return "get_cell_count"
    elif "run_code" in action:
        return "run_code"
    elif (
        "COMPLETE_TASK" in action
        or "Job Submitted" in action
        or action.strip().lower() == "submitted"
    ):
        return "submit"
    else:
        return "other"


def parse_original_notebook_cell(
    cell_data, fallback_index: int | None = None
) -> tuple[int | None, str | None]:
    """Parse one entry from original_notebook into (cell_index, code).

    Supports two formats:
    - Legacy string format: '# --- [CELL 0]: ---\n\n...'
    - Notebook JSON cell dict format: {'cell_type': 'code', 'source': ...}
    """
    # Newer format: real notebook cell dicts from nbformat JSON.
    if isinstance(cell_data, dict):
        if cell_data.get("cell_type") != "code":
            return None, None

        source = cell_data.get("source", "")
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = str(source)

        return fallback_index, code

    # Legacy format: serialized cell text with a header line.
    if isinstance(cell_data, str):
        lines = cell_data.split("\n")
        if not lines or not lines[0].startswith("# --- [CELL "):
            return None, None

        header = lines[0]
        try:
            cell_idx = int(header.split("[CELL ")[1].split("]")[0])
        except (IndexError, ValueError):
            return None, None

        code_lines = lines[1:]
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)

        return cell_idx, "\n".join(code_lines)

    return None, None


def parse_cell_statuses(output: str) -> dict[int, str]:
    """Parse cell execution statuses from run_all output.

    Extracts status information (ok, error, timeout) for each cell.

    Example input:
        Cell 0:
        Execution Count: 1 | Status: ok

        Cell 1:
        Execution Count: 2 | Status: error
        ...

    Returns: {cell_index: status, ...}
    """
    cell_statuses = {}
    lines = output.split("\n")

    current_cell_idx = None
    for line in lines:
        line = line.strip()

        # Check if line starts with "Cell N:"
        if line.startswith("Cell ") and line.endswith(":"):
            try:
                cell_idx = int(line.split("Cell ")[1].rstrip(":"))
                current_cell_idx = cell_idx
            except (IndexError, ValueError):
                pass

        # Check if line contains status information
        elif "Status:" in line and current_cell_idx is not None:
            # Extract status: "Execution Count: 1 | Status: ok" -> "ok"
            parts = line.split("Status:")
            if len(parts) >= 2:
                status = parts[1].strip().split()[0]  # Get first word after Status:
                cell_statuses[current_cell_idx] = status
                current_cell_idx = None

    return cell_statuses


def load_reference_fix_cells(base_dir: Path, instance: str) -> list[str] | None:
    """Load formatted code cells from JunoBench fixed notebook for an instance."""
    if instance in REFERENCE_NOTEBOOK_CACHE:
        return REFERENCE_NOTEBOOK_CACHE[instance]

    fixed_path = (
        base_dir / "JunoBench" / "benchmark" / instance / f"{instance}_fixed.ipynb"
    )
    if not fixed_path.exists():
        REFERENCE_NOTEBOOK_CACHE[instance] = None
        return None

    try:
        nb = load_and_parse_notebook(fixed_path, "JunoBench_Buggy")
        formatted_cells = [
            format_cell_source_for_llm(i, cell)
            for i, cell in enumerate(nb.cells)
            if cell.get("cell_type") == "code"
        ]
        REFERENCE_NOTEBOOK_CACHE[instance] = formatted_cells
        return formatted_cells
    except Exception as e:
        print(
            f"  Warning: Failed to load reference notebook for {instance}: {e}",
            file=sys.stderr,
        )
        REFERENCE_NOTEBOOK_CACHE[instance] = None
        return None


def process_summary(
    summary_file: Path, model: str, library: str, run: str, base_dir: Path
) -> dict | None:
    """Process a single summary JSON file into viewer format."""

    # Extract instance name from filename (e.g., "lightgbm_1_summary.json" -> "lightgbm_1")
    instance = summary_file.stem.replace("_summary", "")

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Error reading {summary_file}: {e}", file=sys.stderr)
        return None

    # Extract metadata
    metadata = summary.get("metadata", {})
    statistics = summary.get("statistics", {})

    # Build steps by combining llm_responses and operations
    llm_responses = summary.get("llm_responses", [])
    operations = summary.get("operations", [])
    code_changes = summary.get("code_changes", [])

    # Create operation lookup by step
    ops_by_step = {
        op.get("step"): op for op in operations if isinstance(op, dict) and "step" in op
    }

    # Create code changes lookup by step
    changes_by_step = {
        change.get("step"): change
        for change in code_changes
        if isinstance(change, dict) and "step" in change
    }

    # Track current state of each cell
    cell_states = {}

    # Parse original notebook to initialize cell states
    original_notebook = summary.get("original_notebook", [])
    for idx, cell_data in enumerate(original_notebook):
        cell_idx, code = parse_original_notebook_cell(cell_data, fallback_index=idx)
        if cell_idx is not None:
            cell_states[cell_idx] = code

    steps = []
    for llm_resp in llm_responses:
        step_num = llm_resp["step"]
        op = ops_by_step.get(step_num, {})

        action = op.get("action", "")
        action_type = classify_action(action)

        # Check if this step involves an edit
        edit_info = None
        change = changes_by_step.get(step_num)
        if change:
            cell_idx = change["cell_index"]
            old_code = cell_states.get(cell_idx, "")
            new_code = change["code"]
            edit_info = {
                "cell_index": cell_idx,
                "old_code": old_code,
                "new_code": new_code,
            }
            cell_states[cell_idx] = new_code

        # Extract cell statuses if this is a run_all operation
        cell_statuses = None
        if action_type == "run_all":
            output = op.get("output", "")
            if output:
                cell_statuses = parse_cell_statuses(output)

        steps.append(
            {
                "step": step_num,
                "assistant": llm_resp.get("content", ""),
                "observation": op.get("output", ""),
                "action": action,
                "action_type": action_type,
                "reasoning": llm_resp.get("reasoning"),
                "edit": edit_info,
                "cell_statuses": cell_statuses,
            }
        )

    # Build notebook_cells with original code
    # The viewer will compute the state at each step using code_changes
    notebook_cells = []
    for idx, cell_data in enumerate(original_notebook):
        cell_idx, original_code = parse_original_notebook_cell(
            cell_data, fallback_index=idx
        )
        if cell_idx is None:
            continue

        notebook_cells.append(
            {
                "index": cell_idx,
                "original_code": original_code,
            }
        )

    # Build a complete code_changes list with step information for the viewer
    code_changes_for_viewer = []
    for change in code_changes:
        code_changes_for_viewer.append(
            {
                "cell_index": change["cell_index"],
                "step": change["step"],
                "code": change["code"],
            }
        )

    # Extract submit reason from the last operation if it's a submission
    submit_reason = None
    if operations:
        last_op = operations[-1]
        last_action = last_op.get("action", "")
        if classify_action(last_action) == "submit":
            # Extract the reason from the action or output
            submit_reason = last_op.get("output", "") or last_action

    # Load reference fixed notebook (already in the same serialized format as original_notebook)
    reference_fix_notebook = load_reference_fix_cells(base_dir, instance)

    return {
        "key": f"{model}/{library}/{run}/{instance}",
        "model": model,
        "library": library,
        "run": run,
        "instance": instance,
        "metadata": {
            "success": metadata.get("success", False),
            "status": metadata.get("status", "UNKNOWN"),
            "cost": metadata.get("cost", 0),
            "execution_time": round(metadata.get("execution_time_seconds", 0), 2),
            "total_steps": statistics.get("total_steps", 0),
            "cells_edited": statistics.get("cells_edited", 0),
            "submit_reason": submit_reason,
        },
        "steps": steps,
        "notebook_cells": notebook_cells,
        "code_changes": code_changes_for_viewer,
        "reference_fix_notebook": reference_fix_notebook,
    }


def scan_trajectories_dir(trajectories_dir: Path, base_dir: Path) -> list[dict]:
    """Scan a trajectories directory for summary files and process them."""
    all_trajectories = []

    if not trajectories_dir.exists():
        return all_trajectories

    # Scan for model directories (format: model_library or just model)
    for model_lib_dir in sorted(trajectories_dir.iterdir()):
        if not model_lib_dir.is_dir():
            continue

        dirname = model_lib_dir.name

        # Skip summary files in root
        if dirname.endswith(".json"):
            continue

        # Parse model and library from directory name
        # Format: glm-4.7-355b_sklearn or glm-4.7-355b_torch or glm-4.7-355b
        parts = dirname.rsplit("_", 1)
        if len(parts) == 2:
            model = parts[0]
            default_library = parts[1]
        else:
            # Single directory name - use as model, derive library from each instance
            model = dirname
            default_library = None

        print(f"Processing {model}...")

        # Scan for run directories
        for run_dir in sorted(model_lib_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            run = run_dir.name
            print(f"  {run}...")

            # Find all summary files in this run
            summary_files = list(run_dir.glob("*_summary.json"))

            for summary_file in sorted(summary_files):
                # Derive library from instance name
                instance = summary_file.stem.replace("_summary", "")

                # Use default_library if set, otherwise extract from instance name
                if default_library:
                    library = default_library
                else:
                    # Extract library from instance name (e.g., sklearn_1 -> sklearn)
                    if "_" in instance:
                        library = instance.rsplit("_", 1)[0]
                    else:
                        library = "unknown"

                result = process_summary(summary_file, model, library, run, base_dir)
                if result:
                    all_trajectories.append(result)
                    print(
                        f"    {instance} ({library}): {len(result['steps'])} steps, success={result['metadata']['success']}"
                    )

    return all_trajectories


def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    output_dir = base_dir / "viewer" / "public"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan all config directories in results/
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        return

    config_dirs = [
        d
        for d in sorted(results_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]

    if not config_dirs:
        print(f"No config directories found in {results_dir}", file=sys.stderr)
        return

    configs_metadata = []

    for config_dir in config_dirs:
        config_name = config_dir.name
        print(f"\n{'='*60}")
        print(f"Processing config: {config_name}")
        print(f"{'='*60}")

        all_trajectories = []

        # Scan the config directory for trajectories
        all_trajectories.extend(scan_trajectories_dir(config_dir, base_dir))

        if all_trajectories:
            # Create config-specific data.json in viewer/public/{config_name}/
            config_output_dir = output_dir / config_name
            config_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = config_output_dir / "data.json"
            output = {"trajectories": all_trajectories}

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            size_mb = os.path.getsize(output_file) / 1024 / 1024
            print(
                f"  ✓ Wrote {len(all_trajectories)} trajectories to {output_file} ({size_mb:.1f} MB)"
            )

            configs_metadata.append(
                {
                    "name": config_name,
                    "trajectory_count": len(all_trajectories),
                    "successful_count": sum(
                        1 for t in all_trajectories if t["metadata"]["success"]
                    ),
                    "file_size_mb": round(size_mb, 1),
                }
            )
        else:
            print(f"  (No trajectories found)")

    # Also create a configs index file for the viewer to list available configs
    configs_index = output_dir / "configs_index.json"
    with open(configs_index, "w", encoding="utf-8") as f:
        json.dump(
            {
                "configs": sorted(configs_metadata, key=lambda x: x["name"]),
                "timestamp": str(Path(__file__).stat().st_mtime),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n{'='*60}")
    print(f"✓ Wrote config index to {configs_index}")
    print(f"✓ Total configs processed: {len(configs_metadata)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
