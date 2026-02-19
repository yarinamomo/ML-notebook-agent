"""Helpers for parsing notebook operation commands and command detection."""

import ast
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class NotebookCommandType(Enum):
    GET_CELL_COUNT = "get_cell_count"
    GET_CELLS = "get_cells"
    GET_CELL = "get_cell"
    EDIT_CELL = "edit_cell"
    RUN_CELL = "run_cell"
    RUN_ALL = "run_all"
    RUN_CUSTOM_CODE = "run_code"


@dataclass
class NotebookCommand:
    kind: NotebookCommandType
    args: tuple[Any, ...]


def parse_notebook_command(command: str) -> NotebookCommand:
    match = re.match(r"^([a-zA-Z_]\w*)\s*\((.*)\)\s*$", command.strip())
    if not match:
        raise ValueError(f"Invalid notebook command format: {command}")

    name = match.group(1)
    args_str = match.group(2).strip()
    command_map = {
        "get_cell_count": NotebookCommandType.GET_CELL_COUNT,
        "get_cells": NotebookCommandType.GET_CELLS,
        "get_cell": NotebookCommandType.GET_CELL,
        "edit_cell": NotebookCommandType.EDIT_CELL,
        "run_cell": NotebookCommandType.RUN_CELL,
        "run_all": NotebookCommandType.RUN_ALL,
        "run_code": NotebookCommandType.RUN_CUSTOM_CODE,
    }
    if name not in command_map:
        raise ValueError(f"Unknown notebook operation: {name}")

    kind = command_map[name]
    if kind in {NotebookCommandType.GET_CELL_COUNT, NotebookCommandType.GET_CELLS, NotebookCommandType.RUN_ALL}:
        if args_str:
            raise ValueError(f"{name}() does not take arguments")
        return NotebookCommand(kind=kind, args=())

    if kind in {NotebookCommandType.GET_CELL, NotebookCommandType.RUN_CELL}:
        if not args_str or "," in args_str:
            raise ValueError(f"{name}() expects a single integer index")
        return NotebookCommand(kind=kind, args=(int(args_str),))

    if kind is NotebookCommandType.EDIT_CELL:
        if not args_str or "," not in args_str:
            raise ValueError("edit_cell() expects index and code string")
        index_str, code_str = args_str.split(",", 1)
        index = int(index_str.strip())
        code = ast.literal_eval(code_str.strip())
        if not isinstance(code, str):
            raise ValueError("edit_cell() code argument must be a string")
        return NotebookCommand(kind=kind, args=(index, code))

    if kind is NotebookCommandType.RUN_CUSTOM_CODE:
        if not args_str:
            raise ValueError("run_code() expects a code string")
        code = ast.literal_eval(args_str.strip())
        if not isinstance(code, str):
            raise ValueError("run_code() code argument must be a code string")
        return NotebookCommand(kind=kind, args=(code,))

    raise ValueError(f"Unsupported notebook operation: {name}")


def strip_markdown_code_blocks(command: str) -> str:
    """Extract the first fenced code block if present; otherwise return original input."""
    stripped = command.strip()
    if "```" not in stripped:
        return stripped

    lines = stripped.split("\n")
    code_start = -1
    code_end = -1

    for i, line in enumerate(lines):
        if line.strip().startswith("```") and code_start == -1:
            code_start = i
        elif line.strip() == "```" and code_start != -1:
            code_end = i
            break

    if code_start != -1 and code_end != -1 and code_end > code_start:
        return "\n".join(lines[code_start + 1:code_end]).strip()

    return stripped


def has_notebook_marker(command: str) -> bool:
    return "__NOTEBOOK_OP__" in command


def is_notebook_command(command: str) -> bool:
    return command.strip().startswith("__NOTEBOOK_OP__")


def is_mixed_notebook_and_bash(command: str) -> bool:
    """
    Check if command chains multiple operations using bash operators.
    
    Returns True if bash operators (&&, ;, |) appear OUTSIDE the notebook 
    operation's argument string, indicating:
    - Notebook + bash: '__NOTEBOOK_OP__run_code("x=1"); echo "hello"'
    - Notebook + notebook: '__NOTEBOOK_OP__run_code("x=1") && __NOTEBOOK_OP__get_cells()'
    - Multiple commands: '__NOTEBOOK_OP__run_all() | grep error'
    
    Returns False for valid single operations:
    - Valid: '__NOTEBOOK_OP__run_code("print(1); print(2)")'  # ; is inside args
    """
    if not has_notebook_marker(command):
        return False
    
    # Extract the part after __NOTEBOOK_OP__
    cmd = command.strip()
    if cmd.startswith("__NOTEBOOK_OP__"):
        cmd = cmd.replace("__NOTEBOOK_OP__", "", 1).strip()
    
    # Check if it matches valid notebook command pattern: operation(args)
    # The $ anchor ensures nothing comes after the closing paren
    match = re.match(r"^([a-zA-Z_]\w*)\s*\((.*)\)\s*$", cmd)
    if match:
        return False  # Valid single notebook operation
    
    # Doesn't match complete pattern - check for chaining operators
    return any(token in command for token in ("&&", ";", "|"))


def extract_notebook_command(command: str) -> str:
    return command.strip().replace("__NOTEBOOK_OP__", "", 1)


def is_bash_command(command: str) -> bool:
    """Check if command looks like a bash command."""
    bash_indicators = [
        "echo ", "cat ", "ls ", "pwd", "cd ", "mkdir ", "rm ", "touch ",
        "grep ", "find ", "sed ", "awk ", "git ", "python ", "pip ",
        "export ", "source ", "./", "bash ", "sh ",
    ]
    command_lower = command.strip().lower()
    return any(command_lower.startswith(indicator) for indicator in bash_indicators)


def wrap_bash_command(command: str) -> str:
    """Wrap bash command in Python subprocess call, matching mini-swe-agent pattern."""
    return f"""
import subprocess

result = subprocess.run(
    {repr(command)},
    shell=True,
    capture_output=True,
    text=True,
    cwd='/app/container',
    encoding='utf-8',
    errors='replace'
)

# Print stdout and stderr
if result.stdout:
    print(result.stdout, end='')
if result.stderr:
    print(result.stderr, end='')

# Print return code marker for parsing
print(f'__RETURNCODE__={{result.returncode}}')
"""
