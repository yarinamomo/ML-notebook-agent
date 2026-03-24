from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class StreamOutput(TypedDict):
    """Stream output (stdout/stderr)."""

    output_type: Literal["stream"]
    name: str
    text: str


class ExecuteResultOutput(TypedDict):
    """Execute result output from code evaluation."""

    output_type: Literal["execute_result"]
    data: Dict[str, Any]
    execution_count: Optional[int]


class DisplayDataOutput(TypedDict):
    """Display data output (plots, tables, etc)."""

    output_type: Literal["display_data"]
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class ErrorOutput(TypedDict):
    """Error output from execution."""

    output_type: Literal["error"]
    ename: str
    evalue: str
    traceback: List[str]


# Union type for all possible outputs
CellOutput = Union[StreamOutput, ExecuteResultOutput, DisplayDataOutput, ErrorOutput]


class CellExecutionResult(TypedDict):
    """Result from executing a notebook cell."""

    execution_count: Optional[int]
    status: str
    done: bool
    outputs: List[CellOutput]


class NotebookCell(TypedDict):
    """Represents a notebook cell."""

    cell_type: Literal["code", "markdown", "raw"]
    execution_count: Optional[int]
    source: str
    metadata: Dict[str, Any]
    outputs: List[CellOutput]
