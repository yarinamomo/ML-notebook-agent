from pathlib import Path
from typing import List, cast
from nbformat import NO_CONVERT, NotebookNode, read
from src.utils.nb_types import CellExecutionResult, NotebookCell


def load_and_parse_notebook(nb_path: Path, parse_mode: str) -> NotebookNode:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = read(f, as_version=NO_CONVERT)
    code_cells: List[NotebookCell] = [
        cell for cell in nb.cells if cell.get("cell_type") == "code"
    ]
    selected = []
    is_buggy_mode = parse_mode == "JunoBench_Buggy"
    buggy_exec_count = _get_buggy_cell_exec_count(nb) if is_buggy_mode else None

    if is_buggy_mode or parse_mode == "JunoBench_Fixed":
        for cell in code_cells:
            exec_count = cell.get("execution_count", None)
            first_line = _first_line(cell)
            is_reexecute = "[reexecute]" in first_line
            is_valid_exec_count = exec_count is not None and (not is_buggy_mode or exec_count <= buggy_exec_count)  # type: ignore[operator]
            if is_reexecute or is_valid_exec_count:
                selected.append(cell)
        nb.cells[:] = sorted(selected, key=lambda cell: cell["execution_count"])
        return nb
    else:
        nb.cells[:] = code_cells
        return nb


def save_cells(
    cells: list[NotebookCell],
    initial_notebook: NotebookNode,
    cell_states: dict[int, str],
    exec_states: dict[int, CellExecutionResult],
    instance_name: str,
    output_dir: Path | None = None,
) -> None:
    """Save the notebook as a Python script with cell metadata."""
    cell_metadata = [
        _get_cell_metadata(cell_states, exec_states, i) for i in range(len(cells))
    ]
    formatted_cells = [
        _format_cell_with_metadata(
            i, cell, cast(NotebookCell, initial_notebook.cells[i]), cell_metadata[i]
        )
        for i, cell in enumerate(cells)
    ]
    script_content = "\n\n#%%\n".join(formatted_cells)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{instance_name}_patched.py"
        save_path.write_text(script_content, encoding="utf-8")


def get_cell_source(cell: NotebookCell) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    return source.strip()


def _get_cell_metadata(
    cell_states: dict[int, str], exec_states: dict[int, CellExecutionResult], index: int
) -> dict:
    return {
        "cell_state": cell_states.get(index, "unchanged"),
        "execution_status": exec_states.get(index, {"status": "not run"}),
    }


def _format_cell_with_metadata(
    index: int, cell: NotebookCell, original_cell: NotebookCell, metadata: dict
) -> str:
    exec_status = metadata["execution_status"]
    exec_status_filtered = (
        {k: v for k, v in exec_status.items() if k != "outputs"}
        if isinstance(exec_status, dict)
        else exec_status
    )

    cell_source = get_cell_source(cell)

    # If the cell has been edited, show BEFORE and AFTER
    if metadata["cell_state"] == "edited" and original_cell:
        original_content = get_cell_source(original_cell)
        commented_original = "\n".join(
            f"# {line}" for line in original_content.splitlines()
        )
        cell_content = f"# === BEFORE (original) ===\n{commented_original}\n\n# === AFTER (edited) ===\n{cell_source}"
    else:
        cell_content = cell_source

    return "\n".join(
        [
            f"# --- [CELL {index}]: ---",
            f"# cell_state: {metadata['cell_state']}",
            f"# execution_status: {exec_status_filtered}",
            cell_content,
        ]
    )


def _first_line(cell: NotebookCell) -> str:
    source = get_cell_source(cell)
    return source.splitlines()[0] if source else ""


def _get_buggy_cell_exec_count(nb: NotebookNode) -> int | None:
    max_exec_count: int | None = None
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        exec_count = cell.get("execution_count")
        if isinstance(exec_count, int):
            max_exec_count = (
                exec_count
                if max_exec_count is None
                else max(max_exec_count, exec_count)
            )

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return cell.get("execution_count")

    # Fixed notebooks can have no error output at all. In that case,
    # keep all executed cells by returning the largest execution count.
    return max_exec_count
