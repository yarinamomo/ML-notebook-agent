"""
Tool definitions and action parsing for notebook-specific LLM tool calls.

Replaces the generic BASH_TOOL with dedicated tools for each notebook operation,
providing structured arguments instead of string-based command parsing.
"""

import json
import logging
import re
import uuid

from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError

logger = logging.getLogger("notebook_tools")

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

GET_CELL_COUNT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_cell_count",
        "description": "Returns the total number of cells in the notebook.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

GET_CELLS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_cells",
        "description": "Returns all cells with their content and index.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

GET_CELL_TOOL = {
    "type": "function",
    "function": {
        "name": "get_cell",
        "description": "Get the source content of a specific cell by its 0-based index.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_index": {
                    "type": "integer",
                    "description": "0-based index of the cell to retrieve.",
                },
            },
            "required": ["cell_index"],
        },
    },
}

EDIT_CELL_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_cell",
        "description": "Replace the entire source content of a cell at the given index with new code.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_index": {
                    "type": "integer",
                    "description": "0-based index of the cell to edit.",
                },
                "code": {
                    "type": "string",
                    "description": "The new Python source code for the cell.",
                },
            },
            "required": ["cell_index", "code"],
        },
    },
}

RUN_CELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_cell",
        "description": "Execute a specific cell by its 0-based index and return its output. Call this when you know the prior cells have been executed.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_index": {
                    "type": "integer",
                    "description": "0-based index of the cell to run.",
                },
            },
            "required": ["cell_index"],
        },
    },
}

RUN_ALL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_all",
        "description": "Run all cells in the notebook sequentially and return their outputs.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

RUN_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": (
            "Execute arbitrary Python code in the notebook kernel for diagnostic purposes "
            "without modifying any cell. Use this to inspect variables, check types, "
            "or run quick experiments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in the notebook kernel.",
                },
            },
            "required": ["code"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Call this tool when all errors have been fixed and the notebook runs successfully. "
            "This submits your final result and ends the session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A brief summary of the fixes you made.",
                },
            },
            "required": ["summary"],
        },
    },
}

# All notebook tools in the order they should be presented
NOTEBOOK_TOOLS = [
    GET_CELL_COUNT_TOOL,
    GET_CELLS_TOOL,
    GET_CELL_TOOL,
    EDIT_CELL_TOOL,
    RUN_CELL_TOOL,
    RUN_ALL_TOOL,
    RUN_CODE_TOOL,
    SUBMIT_TOOL,
]

VALID_TOOL_NAMES = {t["function"]["name"] for t in NOTEBOOK_TOOLS}


# ---------------------------------------------------------------------------
# Action parsing (replaces parse_toolcall_actions for notebook tools)
# ---------------------------------------------------------------------------

def parse_notebook_tool_actions(
    tool_calls: list,
    *,
    format_error_template: str,
    raw_message: dict | None = None,
) -> list[dict]:
    """Parse tool calls from the LLM response into structured action dicts.

    Each returned action has the shape::

        {
            "tool_name": "edit_cell",
            "arguments": {"cell_index": 0, "code": "..."},
            "tool_call_id": "<id>",
            # Legacy key kept for compatibility with observation formatting:
            "command": "edit_cell(cell_index=0, code='...')",
        }

    Raises :class:`FormatError` when the response contains no tool calls or
    references an unknown tool.
    """
    if not tool_calls:
        extra = {"interrupt_type": "FormatError"}
        if raw_message is not None:
            extra["raw_message"] = raw_message
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error=(
                        "No tool calls found in the response. "
                        "Every response MUST include exactly one tool call."
                    ),
                    actions=[],
                ),
                "extra": extra,
            }
        )

    actions: list[dict] = []
    for tool_call in tool_calls:
        name = tool_call.function.name
        error_msg = ""

        # Parse arguments
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}. "
            args = {}

        # Validate tool name
        if name not in VALID_TOOL_NAMES:
            error_msg += (
                f"Unknown tool '{name}'. "
                f"Valid tools are: {', '.join(sorted(VALID_TOOL_NAMES))}."
            )

        if error_msg:
            extra = {"interrupt_type": "FormatError"}
            if raw_message is not None:
                extra["raw_message"] = raw_message
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                    "extra": extra,
                }
            )

        # Build a human-readable "command" string for logging / UI display
        command_repr = _build_command_repr(name, args)

        action = {
            "tool_name": name,
            "arguments": args,
            "command": command_repr,
        }
        if getattr(tool_call, "id", None):
            action["tool_call_id"] = tool_call.id

        actions.append(action)

    return actions


def _build_command_repr(name: str, args: dict) -> str:
    """Build a human-readable representation of a tool call for logging."""
    if not args:
        return f"{name}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 120:
            parts.append(f"{k}=<{len(v)} chars>")
        else:
            parts.append(f"{k}={v!r}")
    return f"{name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Content-based fallback parsing for models that don't use tool_calls
# (e.g. GLM which embeds <tool_call>...</tool_call> in content text)
# ---------------------------------------------------------------------------

# Lightweight mock to satisfy code that accesses tc.function.name / tc.function.arguments / tc.id
class _MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class _MockToolCall:
    def __init__(self, name: str, arguments: str, call_id: str | None = None):
        self.function = _MockFunction(name, arguments)
        self.id = call_id


def parse_tool_calls_from_content(content: str) -> list[_MockToolCall]:
    """Extract tool calls embedded in the LLM's text ``content``.

    Supports several patterns that models (especially GLM) produce:

    1. ``<tool_call>{"name": "run_all", "arguments": {}}</tool_call>``
    2. ``<tool_call>run_all</tool_call>`` (no-arg shorthand)
    3. ``<tool_call>run_all()</tool_call>``
    4. ``<tool_call>edit_cell(cell_index=3, code="...")</tool_call>``
    5. ``<tool_call>{"name": "edit_cell", "arguments": {"cell_index": 3, "code": "..."}}</tool_call>``

    Returns a (possibly empty) list of ``_MockToolCall`` objects compatible
    with ``parse_notebook_tool_actions``.
    """
    if not content:
        return []

    calls: list[_MockToolCall] = []

    # Pattern: <tool_call>...</tool_call>  (greedy-minimal match)
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        inner = match.group(1).strip()
        parsed = _parse_single_tool_call_content(inner)
        if parsed:
            calls.append(parsed)

    # Strict mode: only parse explicit <tool_call> blocks. This avoids
    # executing tools from incidental prose such as "I'll run_all() next".
    return calls


def _parse_single_tool_call_content(inner: str) -> _MockToolCall | None:
    """Parse the content between ``<tool_call>`` tags."""
    # Try JSON first: {"name": "...", "arguments": {...}}
    try:
        obj = json.loads(inner)
        if isinstance(obj, dict) and "name" in obj:
            name = obj["name"]
            arguments = obj.get("arguments", obj.get("parameters", {}))
            return _MockToolCall(name, json.dumps(arguments))
    except (json.JSONDecodeError, TypeError):
        pass

    # Bare tool name: "run_all" or "get_cells"
    bare = inner.strip().rstrip("()")
    if bare in VALID_TOOL_NAMES:
        return _MockToolCall(bare, "{}")

    # Function-call style: "edit_cell(0, \"code\")" or "run_cell(cell_index=3)"
    fn_match = re.match(r"^([a-zA-Z_]\w*)\s*\((.*)\)$", inner, re.DOTALL)
    if fn_match:
        name = fn_match.group(1)
        if name in VALID_TOOL_NAMES:
            args = _parse_positional_args(name, fn_match.group(2).strip())
            if args is not None:
                return _MockToolCall(name, json.dumps(args))

    return None


# Maps tool name → ordered list of parameter names for positional arg parsing
_TOOL_PARAM_ORDER: dict[str, list[str]] = {
    "get_cell_count": [],
    "get_cells": [],
    "get_cell": ["cell_index"],
    "edit_cell": ["cell_index", "code"],
    "run_cell": ["cell_index"],
    "run_all": [],
    "run_code": ["code"],
    "submit": ["summary"],
}


def _parse_positional_args(tool_name: str, args_str: str) -> dict | None:
    """Parse a function-call argument string into a dict.

    Handles both keyword (``cell_index=3``) and positional (``3, "code"``)
    styles, mapping positional args to the appropriate parameter names.
    """
    param_names = _TOOL_PARAM_ORDER.get(tool_name)
    if param_names is None:
        return None

    if not args_str:
        return {}

    # Try keyword style first: cell_index=3, code="..."
    kw_match = re.match(r'^([a-zA-Z_]\w*)\s*=', args_str)
    if kw_match:
        # Use a simple parser for key=value pairs
        result = {}
        # Split carefully (respecting strings)
        try:
            # Wrap in dict() syntax and eval safely
            result = _safe_parse_kwargs(args_str)
            return result
        except Exception:
            pass

    # Positional: split on first comma for 2-arg tools, otherwise whole string
    if len(param_names) == 0:
        return {}
    elif len(param_names) == 1:
        val = _try_parse_value(args_str)
        return {param_names[0]: val}
    elif len(param_names) == 2:
        # Split on first comma that isn't inside quotes
        first, rest = _split_first_arg(args_str)
        if first is not None:
            return {
                param_names[0]: _try_parse_value(first),
                param_names[1]: _try_parse_value(rest),
            }

    return None


def _try_parse_value(s: str):
    """Parse a string as int, float, or string literal."""
    s = s.strip()
    # Try int
    try:
        return int(s)
    except ValueError:
        pass
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    # Try JSON string ("..." or '...')
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try Python string literal
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # Return as-is
    return s


def _split_first_arg(s: str) -> tuple[str | None, str]:
    """Split on the first comma not inside quotes/parens."""
    depth = 0
    in_str: str | None = None
    escape = False
    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if in_str:
            if c == in_str:
                in_str = None
            continue
        if c in ('"', "'"):
            in_str = c
            continue
        if c in ('(', '[', '{'):
            depth += 1
        elif c in (')', ']', '}'):
            depth -= 1
        elif c == ',' and depth == 0:
            return s[:i].strip(), s[i + 1:].strip()
    return None, s


def _safe_parse_kwargs(s: str) -> dict:
    """Parse 'key1=val1, key2=val2' into a dict.

    Uses a regex approach to extract key=value pairs, handling quoted strings.
    """
    result = {}
    # Match key="value" or key='value' or key=number patterns
    # Use a pattern that handles quoted strings with commas inside
    remaining = s.strip()
    while remaining:
        # Match: identifier = value
        kv_match = re.match(r'^([a-zA-Z_]\w*)\s*=\s*', remaining)
        if not kv_match:
            break
        key = kv_match.group(1)
        remaining = remaining[kv_match.end():]
        # Parse the value
        if remaining.startswith('"'):
            # Double-quoted string
            end = _find_closing_quote(remaining, '"')
            result[key] = remaining[1:end]
            remaining = remaining[end + 1:].lstrip(', ')
        elif remaining.startswith("'"):
            # Single-quoted string
            end = _find_closing_quote(remaining, "'")
            result[key] = remaining[1:end]
            remaining = remaining[end + 1:].lstrip(', ')
        else:
            # Non-string value: take until comma or end
            comma_pos = remaining.find(',')
            if comma_pos == -1:
                val_str = remaining.strip()
                remaining = ''
            else:
                val_str = remaining[:comma_pos].strip()
                remaining = remaining[comma_pos + 1:].strip()
            result[key] = _try_parse_value(val_str)
    return result


def _find_closing_quote(s: str, quote_char: str) -> int:
    """Find the index of the closing quote, handling escapes."""
    i = 1  # skip opening quote
    while i < len(s):
        if s[i] == '\\':
            i += 2
            continue
        if s[i] == quote_char:
            return i
        i += 1
    return len(s) - 1  # fallback: end of string

