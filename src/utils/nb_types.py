from typing import TypedDict, List


class CellOutput(TypedDict):
    output_type: str
    name: str
    text: str

class CellExecutionResult(TypedDict):
    execution_count: int
    outputs: List[CellOutput]
    status: str