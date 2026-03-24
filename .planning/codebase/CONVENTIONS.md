# Coding Conventions

**Analysis Date:** 2026-03-24

## Naming Patterns

**Files:**
- `snake_case.py` for module files (e.g., `notebook_tools.py`, `format_nb_cells.py`, `yaml_parser.py`)
- Test files: `test_<module>_<type>.py` (e.g., `test_sandbox_unit.py`, `test_sandbox_integration.py`)
- Private helper modules: prefixed with underscore (e.g., `_private_helper.py`)

**Functions:**
- `snake_case` for all functions (e.g., `parse_notebook_tool_actions`, `get_instance_trajectory_path`)
- Private functions: `_snake_case` (e.g., `_run_cell`, `_on_ws_message`, `_start_kernel`)
- Factory functions: `create_<noun>` or `make_<noun>` pattern

**Classes:**
- `PascalCase` for all classes (e.g., `DockerSandbox`, `NotebookEnvironment`, `UiAgent`, `BenchmarkProblem`)
- Exceptions: `PascalCase` with descriptive names (e.g., `AgentTimeout`, `EnvironmentUnavailable`)
- Abstract base classes and interfaces follow same pattern

**Variables:**
- `snake_case` for all variables (e.g., `execution_results`, `kernel_id`, `cell_index`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `UI_ENABLED`, `NOTEBOOK_TOOLS`, `VALID_TOOL_NAMES`)
- Private instance variables: `_snake_case` (e.g., `_start_time`, `_cells`)
- Type variables: `T`, `T_Generic` following PEP 484

**Types:**
- `PascalCase` for TypedDict and dataclass types (e.g., `CellExecutionResult`, `StreamOutput`, `NotebookCell`)
- Use `TypedDict` for structured data with known fields
- Use `dataclass` for mutable data objects with methods

## Code Style

**Formatting:**
- Use Black or equivalent formatter
- Line length: 88 characters (Black default)
- Use explicit return statements

**Type Hints:**
- Required on all function parameters and return values
- Use `Optional[T]` instead of `T | None` for consistency with pre-3.10 compatibility
- Import from `typing` module (e.g., `from typing import Any, Optional, TypedDict, List`)
- Use `TYPE_CHECKING` block for type-only imports to avoid circular dependencies
- Annotate internal private modules with `if TYPE_CHECKING` for forward references

**Linting:**
- No explicit linter configuration found (`.eslintrc`, `.flake8`, etc.)
- Follow PEP 8 guidelines
- Use descriptive variable and function names

## Import Organization

**Order:**
1. Standard library imports
2. Third-party imports
3. Local application imports (from `src.`)
4. Relative imports (only within packages)

**Grouping:**
- Separate each group with a blank line
- Sort imports alphabetically within groups
- Use `from module import SpecificClass` for specific imports
- Use `from module import *` only for cases like `__all__` definitions

**Path Aliases:**
- `src.` prefix is used for local imports (e.g., `from src.utils.log import logger`)
- No configured path aliases in `pyproject.toml` or `pytest.ini`
- Use absolute imports from `src` over relative imports

**Type-checking Imports:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sandbox import DockerSandbox
```

## Error Handling

**Patterns:**
- Use `logger.exception()` for errors with traceback: `logger.exception("Error executing tool", exc_info=True)`
- Use `logger.error()` for errors without traceback: `logger.error("Failed to load config")`
- Use `logger.warning()` for recoverable issues
- Raise custom exceptions defined in each module
- Use structured error messages with context

**Logging Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General information about program execution
- `WARNING`: Something unexpected happened
- `ERROR`: A serious problem occurred
- Environment variable: `NOTEBOOK_AGENT_LOG_LEVEL` controls level (default: ERROR)

**Custom Exceptions:**
- Inherit from appropriate base exceptions (e.g., `InterruptAgentFlow` from mini-swe-agent)
- Define exceptions at module level near class definitions that use them
- Include helpful error messages in exception data

## Logging

**Framework:** Rich (`rich.logging.RichHandler`)

**Logger Setup:**
- Logger name: `"mlnotebookagent"`
- Configuration in `src/utils/log.py`
- Use `logger` imported from `src.utils.log`
- Logging supports file output via `add_file_handler()`

**Patterns:**
```python
from src.utils.log import logger

logger.info("Starting agent execution")
logger.warning("WebSocket disconnected, attempting to reconnect")
logger.error("Failed to parse tool call")
logger.exception("Kernel execution failed", exc_info=True)
```

**Log Format in File:**
```
[threadName][asctime] - [name] - [levelname] - [message]
```

## Comments

**When to Comment:**
- Docstrings for all modules, classes, and public functions
- Comments for complex logic or algorithms
- Comments explaining "why" not "what"
- TODO comments in format: `# TODO <description>`

**JSDoc/TSDoc:**
- Not applicable (Python-only codebase)
- Use standard Python docstrings (triple quotes)

**Docstring Format:**
```python
def parse_notebook_tool_actions(tool_calls: list, *, format_error_template: str) -> list[dict]:
    """Parse tool calls from the LLM response into structured action dicts.

    Each returned action has the shape::

        {
            "tool_name": "edit_cell",
            "arguments": {"cell_index": 0, "code": "..."},
            "tool_call_id": "<id>",
            "command": "edit_cell(cell_index=0, code='...')",
        }

    Raises:
        FormatError: When response contains no tool calls or unknown tool

    Args:
        tool_calls: List of tool call objects from LLM response
        format_error_template: Error message template for FormatError

    Returns:
        List of structured action dictionaries
    """
```

## Function Design

**Size:**
- Keep functions focused on single responsibility
- Large functions (100+ lines) should be split into smaller helpers
- Use private functions for complex sub-operations

**Parameters:**
- Use keyword arguments for clarity: `def run(code: str, timeout: int = 30)`
- Use `*` to force keyword-only arguments: `def parse(*, format_error_template: str)`
- Type hint all parameters
- Use `Optional[T]` for nullable parameters
- Use `Any` sparingly, prefer specific types

**Return Values:**
- Always annotate return type
- Return `None` for functions that don't produce meaningful output
- Use TypedDict or dataclass for complex return values
- Early returns for error conditions

## Module Design

**Exports:**
- Explicitly specify public API using `__all__` lists
- Keep `__all__` near top of module after imports
- Private functions don't go in `__all__`

**Barrel Files:**
- Location: Package-level `__init__.py` files
- Usage: Not extensively used; most imports are direct

**Constants:**
- Define module-level constants at top after docstring and imports
- Group related constants together

**Module Structure:**
```python
"""Module docstring."""

# Imports
from standard_library import ...
from third_party import ...
from local import ...

# Constants
CONSTANT_VALE = ...

# Type definitions
class MyClass(TypedDict):
    ...

# Private helpers
def _helper():
    ...

# Public API
def public_function():
    ...

# __all__
__all__ = ["public_function", "MyClass"]
```

## Threading and Concurrency

**Thread Safety:**
- Use locks for shared state: `with self.ws_lock:`
- Log thread names: `%(threadName)s` in log format
- Use context managers for thread pools: `with ThreadPoolExecutor() as executor:`

## Context Managers

**Patterns:**
- Use `@contextmanager` for resource cleanup (e.g., `wait()`, `wait_llm()`)
- Implement `__enter__` and `__exit__` for custom context managers
- Always clean up resources in `finally` blocks

## Testing Patterns in Source Code

**Mocking for Testing:**
- Use dependency injection for testability
- Provide factory functions for test substitutions
- Use patch decorators sparingly in production code

---

*Convention analysis: 2026-03-24*