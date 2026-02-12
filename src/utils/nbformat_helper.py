from pathlib import Path
from typing import List
from nbformat import NO_CONVERT, NotebookNode, read, write
from src.utils.nb_types import CellExecutionResult

def load_notebook(nb_path: Path) -> NotebookNode:
    with open(nb_path, "r", encoding="utf-8") as f:
        return read(f, as_version=NO_CONVERT)
    
def get_cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    return source

def select_code_cells(nb: NotebookNode, parse_mode: str) -> List[dict]:
    code_cells = _get_code_cells(nb)
    selected = []
    is_buggy_mode = parse_mode == "JunoBench_Buggy"
    buggy_exec_count = _get_buggy_cell_exec_count(code_cells) if is_buggy_mode else None

    if is_buggy_mode or parse_mode == "JunoBench_Fixed":
        for cell in code_cells:
            exec_count = cell.get("execution_count")
            first_line = _first_line(cell)
            is_reexecute = "[reexecute]" in first_line
            is_valid_exec_count = exec_count is not None and (not is_buggy_mode or exec_count <= buggy_exec_count)
            if is_reexecute or is_valid_exec_count:
                selected.append(cell)
        return sorted(selected, key=lambda cell: cell["execution_count"])
    else:
        return _get_code_cells(nb)  # return all code cells without specific ordering


def save_cells(cells: list[dict], cell_states: dict[int, str], exec_states: dict[int, CellExecutionResult], problem_file: Path) -> None:
    """Save the notebook as a Python script with cell metadata."""
    cell_metadata = [
        _get_cell_metadata(cell_states, exec_states, i) for i in range(len(cells))
    ]
    formatted_cells = [_format_cell_with_metadata(i, cell, cell_metadata[i]) for i, cell in enumerate(cells)]
    script_content = "\n\n#%%\n".join(formatted_cells)
    save_path = (problem_file.with_suffix("") # remove .ipynb suffix
                 .with_name(problem_file.stem + "_patched.py"))
    save_path.write_text(script_content, encoding="utf-8")


def _get_cell_metadata(cell_states: dict[int, str], exec_states: dict[int, CellExecutionResult], index: int) -> dict:
    return {
        "cell_state": cell_states.get(index, "unchanged"), 
        "execution_status": exec_states.get(index, {"status": "not run"})
        }



def _format_cell_with_metadata(index: int, cell: dict, metadata: dict) -> str:
    return "\n".join([
        f"# --- [CELL {index}]: ---",
        f"# cell_state: {metadata['cell_state']}",
        f"# execution_status: {metadata['execution_status']}",
        get_cell_source(cell),
    ])


def _get_code_cells(nb: NotebookNode) -> List[dict]:
    return [cell for cell in nb.cells if cell.get("cell_type") == "code"]

def _first_line(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    source = source.strip()
    return source.splitlines()[0] if source else ""


def _get_buggy_cell_exec_count(code_cells: List[dict]) -> int | None:
    buggy_index = -1

    for idx, cell in enumerate(code_cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                buggy_index = idx
                break

    if buggy_index == -1:
        raise ValueError("No error output found in the notebook.")
    return code_cells[buggy_index].get("execution_count")


