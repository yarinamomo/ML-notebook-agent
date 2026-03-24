"""
Summary utility for generating execution summaries from agent messages.
Parses the message array to extract LLM responses, operations, and code changes.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def extract_reasoning(msg: dict) -> str:
    """Extract reasoning content from an assistant message.

    Checks ``msg["reasoning_content"]`` first, then falls back to
    ``msg["extra"]["response"]["choices"][0]["message"]["reasoning_content"]``.
    """
    reasoning = msg.get("reasoning_content", "")
    if not reasoning:
        llm_resp = msg.get("extra", {}).get("response", {})
        choices = llm_resp.get("choices", [])
        if choices:
            reasoning = choices[0].get("message", {}).get("reasoning_content", "") or ""
    return reasoning


def find_action(actions: list[dict], tool_call_id: str) -> dict | None:
    """Find the action whose ``tool_call_id`` matches *tool_call_id*."""
    for action in actions:
        if action.get("tool_call_id", "") == tool_call_id:
            return action
    return None


def build_summary(
    messages: list[dict],
    cost: float | None,
    execution_time_seconds: float,
    initial_cells: list[str] | None,
) -> dict[str, Any]:
    llm_responses: list[dict] = []
    operations: list[dict] = []
    code_changes: list[dict] = []
    exit_status: str = ""

    last_assistant: dict = {}
    step = 0
    for idx, msg in enumerate(messages):
        role = msg.get("role", "")

        if role == "assistant":
            step += 1
            last_assistant = msg
            content = msg.get("content", "") or ""
            reasoning = extract_reasoning(msg)

            llm_responses.append(
                {
                    "step": step,
                    "content": content,
                    "reasoning": reasoning,
                }
            )

            # Track edit_cell actions for code-change tracking
            for action in msg.get("extra", {}).get("actions", []):
                if action.get("tool_name") == "edit_cell":
                    args = action.get("arguments", {})
                    cell_index = int(args.get("cell_index", -1))
                    code = args.get("code", "")
                    if cell_index >= 0:
                        code_changes.append(
                            {"cell_index": cell_index, "step": step, "code": code}
                        )

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            extra = msg.get("extra", {})
            raw_output = extra.get("raw_output", "")
            returncode = extra.get("returncode", 0)

            # Find the matching command from the preceding assistant message
            matched = find_action(
                last_assistant.get("extra", {}).get("actions", []), tool_call_id
            )
            command = matched.get("command", "") if matched else ""

            # Calculate tool execution time
            execution_time = None
            if (
                last_assistant
                and msg.get("extra", {}).get("timestamp")
                and last_assistant.get("extra", {}).get("timestamp")
            ):
                obs_timestamp = msg.get("extra", {}).get("timestamp")
                llm_timestamp = last_assistant.get("extra", {}).get("timestamp")
                execution_time = round(obs_timestamp - llm_timestamp, 4)

            operation = {
                "step": step,
                "action": command,
                "output": raw_output,
                "return_code": returncode,
                "success": returncode == 0,
            }
            if execution_time is not None:
                operation["execution_time_seconds"] = execution_time

            operations.append(operation)

        elif role == "exit":
            extra = msg.get("extra", {})
            exit_status = extra.get("exit_status", "")
            submission = extra.get("submission", "")
            operations.append(
                {
                    "step": step,
                    "action": exit_status,
                    "output": submission,
                    "return_code": 0 if exit_status == "Submitted" else 1,
                    "success": exit_status in ["Submitted", "Success"],
                }
            )

    # Build code changes list (each edit as a separate entry with step)
    unique_cells_edited = len({c["cell_index"] for c in code_changes})
    successful_ops = sum(1 for op in operations if op.get("success", False))

    # Calculate average tool execution time
    execution_times = [
        float(op["execution_time_seconds"])
        for op in operations
        if op.get("execution_time_seconds") is not None
    ]
    avg_execution_time = (
        round(sum(execution_times) / len(execution_times), 2)
        if execution_times
        else 0.0
    )

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time_seconds, 2),
            "success": exit_status in ["Submitted", "Success"],
            "status": exit_status or "INCOMPLETE",
            "cost": round(cost, 4) if cost is not None else None,
        },
        "statistics": {
            "total_steps": len(llm_responses),
            "total_operations": len(operations),
            "successful_operations": successful_ops,
            "failed_operations": len(operations) - successful_ops,
            "cells_edited": len(code_changes),
            "unique_cells_edited": unique_cells_edited,
            "avg_tool_execution_time_seconds": avg_execution_time,
        },
        "original_notebook": initial_cells or [],
        "llm_responses": llm_responses,
        "operations": operations,
        "code_changes": code_changes,
    }
