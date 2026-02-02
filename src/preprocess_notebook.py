import os
import re
from typing import Dict, List, Optional, Tuple
from nbformat import NO_CONVERT, read
import copy
import tokenize
from io import StringIO
from pathlib import Path

def parse_nb(nb_path: Path, parse_mode = "JunoBench_Buggy") -> List[str]:
    print(f"Parsing notebook at: {nb_path} with mode: {parse_mode}")
    # return with all comments removed from the code cells
    if parse_mode == "JunoBench_Buggy":
        processed_nb = preprocess_junobench_buggy_code_cells(nb_path)
        sort_key = "execution_count"
    elif parse_mode == "JunoBench_Fixed":
        processed_nb = preprocess_auto_executed_code_cells(nb_path)
        sort_key = "execution_count"
    else: # parse_mode == "All_Code_Cells":
        processed_nb = preprocess_all_code_cells(nb_path)
        sort_key = "code_cell_id"

    processed_nb = sorted(processed_nb, key=lambda cell: cell[sort_key])

    res = []
    for exec_item in processed_nb:
        res.append(remove_comments(exec_item["code"]))
    return res

# process reproduced crashing notebooks to: a list of successfully executed code cells + crashing code cell
def preprocess_junobench_buggy_code_cells(nb_path: Path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = read(f, as_version=NO_CONVERT)

    # print(f"Processing notebook: {nb_path}")

    id_name = nb_path.name.replace("_reproduced.ipynb","")
    buggy_index, _ = _find_buggy_cell_index_and_line(nb.cells)
    if buggy_index is None:
        raise ValueError(f"No error output found in the notebook {id_name}.")
    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    buggy_exec_count = code_cells[buggy_index].get('execution_count')
    
    processed_nb = []
    code_cell_count = 0  # Track code cells for logical indexing
    for cell in code_cells:
        exec_count = cell.get('execution_count')
        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
        # print(code_cell_count, buggy_index, exec_count, buggy_exec_count)
        if ("[reexecute]" in first_line):
            processed_nb.append({
                "execution_count": exec_count, 
                "code_cell_id": code_cell_count, 
                "code": cell.source})
        if ((exec_count is not None) and (exec_count <= buggy_exec_count)):
            processed_nb.append({
                "execution_count": exec_count, 
                "code_cell_id": code_cell_count, 
                "code": cell.source})
        code_cell_count += 1

    if not processed_nb:
        print(f"No target cell assigned to {id_name}!")
    if len(processed_nb)<=0:
        print(f"No executed cell(s) assigned to {id_name}!")

    # print(f"Total {len(processed_nb)} code cells processed for notebook: {id_name}")
    return processed_nb

# for any notebook: only get the code cells that have been executed
def preprocess_auto_executed_code_cells(nb_fix_path: Path):
    with open(nb_fix_path, 'r', encoding='utf-8') as f:
        nb_fix = read(f, as_version=NO_CONVERT)

    processed_nb = []
    code_cell_count = 0  # Track code cells for logical indexing
    code_cells = [cell for cell in nb_fix.cells if cell.get("cell_type") == "code"]

    for cell in code_cells:
        exec_count = cell.get('execution_count')
        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
        if ("[reexecute]" in first_line):
            processed_nb.append({
                "execution_count": exec_count, 
                "code_cell_id": code_cell_count, 
                "code": cell.source})
        if (exec_count is not None):
            processed_nb.append({
                "execution_count": exec_count, 
                "code_cell_id": code_cell_count, 
                "code": cell.source})
        code_cell_count += 1

    return processed_nb

# for any notebook: get all code cells
def preprocess_all_code_cells(nb_path: Path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = read(f, as_version=NO_CONVERT)

    processed_nb = []
    code_cell_count = 0  # Track code cells for logical indexing
    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    for cell in code_cells:
        exec_count = cell.get('execution_count')
        processed_nb.append({
            "execution_count": exec_count, 
            "code_cell_id": code_cell_count, 
            "code": cell.source})
        code_cell_count += 1

    return processed_nb

def parse_traceback(str_traceback: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', str_traceback)

def remove_comments(source_code: str) -> str:
    tokens = tokenize.generate_tokens(StringIO(source_code).readline)
    result = []
    last_lineno = -1
    last_col = 0

    for token in tokens:
        tok_type = token.type
        tok_string = token.string
        start_line, start_col = token.start

        if tok_type == tokenize.COMMENT:
            continue

        if start_line > last_lineno:
            result.append('\n' * (start_line - last_lineno - 1))
            last_col = 0

        if start_col > last_col:
            result.append(' ' * (start_col - last_col))

        result.append(tok_string)
        last_lineno, last_col = token.end

    cleaned_code = ''.join(result)
    # Optional: strip trailing spaces from each line
    cleaned_code = '\n'.join(line.rstrip() for line in cleaned_code.splitlines())
    return cleaned_code #normalize_whitespace(cleaned_code)

def _extract_bug_location_from_cell(cell: dict) -> Optional[Tuple[Optional[int], Optional[str]]]:
    """
    Extract the crashing line number from the error traceback in a code cell's output.
    Returns a 1-based line number and line of code or None if not found.
    """
    if 'outputs' not in cell:
        return None, None
    
    for output in cell['outputs']:
        if output.output_type == 'error':
            traceback_lines = output.get('traceback', [])
            traceback_lines = parse_traceback("\n".join(traceback_lines))
            pattern = r'<ipython-input-(\d+)-[\da-f]+> in <cell line: (\d+)>()'
            match = re.search(pattern, traceback_lines)
            if match:
                line_number = int(match.group(2))
                source_lines = cell.get("source", [])
                if isinstance(source_lines, str):
                    source_lines = source_lines.splitlines()
                if 1 <= line_number <= len(source_lines):
                    return line_number, source_lines[line_number - 1].strip()
    return None, None

def _find_buggy_cell_index_and_line(nb_cells: List[dict]) -> Tuple[Optional[int], Optional[int]]:
    """
    Identify the buggy code cell index and the crashing line number.
    """
    code_cell_index = 0
    for cell in nb_cells:
        if cell.cell_type != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                line = _extract_bug_location_from_cell(cell)
                return code_cell_index, line
        code_cell_index += 1
    return None, None