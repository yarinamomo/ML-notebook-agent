#!/usr/bin/env python3
"""
Preprocess trajectory data into a single JSON file for the viewer app.
Reads all trajectory files from the trajectories/ directory and writes
viewer/public/data.json.
"""

import json
import os
import re
import ast
import sys
from pathlib import Path


def parse_patched_py(filepath: str) -> list[dict]:
    """Parse a _patched.py file into a list of notebook cells."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by cell separator
    raw_cells = re.split(r"#%%\n", content)
    cells = []

    for raw in raw_cells:
        raw = raw.strip()
        if not raw:
            continue

        # Extract cell header
        header_match = re.match(
            r"# --- \[CELL (\d+)\]: ---\n"
            r"# cell_state: (\w+)\n"
            r"# execution_status: (.+?)\n",
            raw,
        )
        if not header_match:
            continue

        cell_index = int(header_match.group(1))
        cell_state = header_match.group(2)
        exec_status_str = header_match.group(3)

        # Parse execution status
        try:
            exec_status = ast.literal_eval(exec_status_str)
        except Exception:
            exec_status = {"status": "unknown"}

        body = raw[header_match.end() :]

        original_code = None
        code = body

        if cell_state == "edited":
            # Extract BEFORE and AFTER sections
            before_match = re.search(
                r"# === BEFORE \(original\) ===\n(.*?)# === AFTER \(edited\) ===\n",
                body,
                re.DOTALL,
            )
            if before_match:
                before_block = before_match.group(1)
                # Remove leading "# " from each line
                original_lines = []
                for line in before_block.split("\n"):
                    if line.startswith("# "):
                        original_lines.append(line[2:])
                    elif line == "#":
                        original_lines.append("")
                    elif line.strip() == "":
                        continue
                    else:
                        original_lines.append(line)
                original_code = "\n".join(original_lines).strip()

                # The AFTER section is everything after the marker
                after_start = before_match.end()
                code = body[after_start:].strip()
            else:
                code = body.strip()
        else:
            code = body.strip()

        cells.append(
            {
                "index": cell_index,
                "state": cell_state,
                "code": code,
                "original_code": original_code,
                "execution_status": exec_status.get("status", "unknown"),
            }
        )

    return cells


def parse_edit_action(action: str) -> dict | None:
    """Parse an __NOTEBOOK_OP__edit_cell action to extract cell index and new code."""
    match = re.match(
        r"__NOTEBOOK_OP__edit_cell\((\d+),\s*(.*)\)\s*$", action, re.DOTALL
    )
    if not match:
        return None

    cell_idx = int(match.group(1))
    code_str = match.group(2).strip()

    # Try to parse the string argument
    try:
        code = ast.literal_eval(code_str)
    except Exception:
        # Fallback: try simple unescape
        if code_str.startswith('"') and code_str.endswith('"'):
            code = code_str[1:-1].replace("\\n", "\n").replace("\\t", "\t").replace(
                '\\"', '"'
            )
        else:
            code = code_str

    return {"cell_index": cell_idx, "new_code": code}


def classify_action(action: str) -> str:
    """Classify the action type."""
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
    elif "COMPLETE_TASK" in action or "Job Submitted" in action:
        return "submit"
    else:
        return "other"


def process_trajectory(traj_dir: str, model: str, library: str, run: str, instance: str) -> dict | None:
    """Process a single trajectory (one instance)."""
    traj_file = os.path.join(traj_dir, f"{instance}.traj.json")
    summary_file = os.path.join(traj_dir, f"{instance}_summary.json")
    patched_file = os.path.join(traj_dir, f"{instance}_patched.py")

    if not all(os.path.exists(f) for f in [traj_file, summary_file, patched_file]):
        print(f"  Skipping {instance}: missing files", file=sys.stderr)
        return None

    # Read files
    with open(traj_file, "r", encoding="utf-8") as f:
        traj_data = json.load(f)
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # Parse notebook cells from patched file
    notebook_cells = parse_patched_py(patched_file)

    # Build original code map from code_changes
    original_code_map = {}
    for change in summary_data.get("code_changes", []):
        cell_id = change["cell_id"]
        original_code_map[cell_id] = change["original_code"].strip()

    # Also build from notebook cells
    for cell in notebook_cells:
        if cell["original_code"] is not None and cell["index"] not in original_code_map:
            original_code_map[cell["index"]] = cell["original_code"]

    # Process messages into steps
    messages = traj_data.get("messages", [])
    operations = summary_data.get("operations", [])

    steps = []
    cell_current_state = dict(original_code_map)  # Track current state of each cell

    # Build steps from messages
    # Pattern: system, user, [assistant, user, assistant, user, ...]
    step_num = 0
    i = 2  # Skip system and first user message
    while i < len(messages):
        msg = messages[i]
        if msg["role"] != "assistant":
            i += 1
            continue

        step_num += 1
        assistant_content = msg.get("content", "")

        # Get reasoning/thinking if available
        reasoning = None
        extra = msg.get("extra", {})
        if extra:
            resp = extra.get("response", {})
            choices = resp.get("choices", [])
            if choices:
                inner_msg = choices[0].get("message", {})
                reasoning = inner_msg.get("reasoning_content")

        # Get observation (next user message)
        observation = ""
        if i + 1 < len(messages) and messages[i + 1]["role"] == "user":
            observation = messages[i + 1].get("content", "")

        # Find matching operation
        op = None
        for o in operations:
            if o["step"] == step_num:
                op = o
                break

        action = op["action"] if op else ""
        action_type = classify_action(action)

        # Check for edit
        edit_info = None
        if action_type == "edit_cell":
            parsed = parse_edit_action(action)
            if parsed:
                cell_idx = parsed["cell_index"]
                new_code = parsed["new_code"]
                old_code = cell_current_state.get(cell_idx, "")
                edit_info = {
                    "cell_index": cell_idx,
                    "old_code": old_code,
                    "new_code": new_code,
                }
                cell_current_state[cell_idx] = new_code

        steps.append(
            {
                "step": step_num,
                "assistant": assistant_content,
                "observation": observation,
                "action": action,
                "action_type": action_type,
                "reasoning": reasoning,
                "edit": edit_info,
            }
        )

        i += 2  # Skip to next assistant message

    # Metadata
    metadata = summary_data.get("metadata", {})

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
            "total_steps": summary_data.get("statistics", {}).get("total_steps", 0),
            "cells_edited": summary_data.get("statistics", {}).get("cells_edited", 0),
        },
        "steps": steps,
        "notebook_cells": notebook_cells,
    }


def main():
    trajectories_dir = Path(__file__).parent / "trajectories"
    output_dir = Path(__file__).parent / "viewer" / "public"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "data.json"

    all_trajectories = []

    # Scan for model directories (format: model_library)
    for model_lib_dir in sorted(trajectories_dir.iterdir()):
        if not model_lib_dir.is_dir():
            continue

        dirname = model_lib_dir.name
        # Parse model and library from directory name
        # Format: glm-4.7-355b_sklearn or glm-4.7-355b_torch
        parts = dirname.rsplit("_", 1)
        if len(parts) != 2:
            continue

        model = parts[0]
        library = parts[1]

        print(f"Processing {model}/{library}...")

        # Scan for run directories
        for run_dir in sorted(model_lib_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            run = run_dir.name
            print(f"  {run}...")

            # Find all instances in this run
            instances = set()
            for f in run_dir.iterdir():
                if f.name.endswith(".traj.json"):
                    inst = f.name.replace(".traj.json", "")
                    instances.add(inst)

            for instance in sorted(instances):
                result = process_trajectory(
                    str(run_dir), model, library, run, instance
                )
                if result:
                    all_trajectories.append(result)
                    print(f"    {instance}: {len(result['steps'])} steps, success={result['metadata']['success']}")

    # Write output
    output = {"trajectories": all_trajectories}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    # Print size
    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\nWrote {len(all_trajectories)} trajectories to {output_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
