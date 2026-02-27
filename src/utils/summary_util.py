"""
Summary utility for generating execution summaries from agent messages.
Parses the message array to extract LLM responses, operations, and code changes.
"""
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from minisweagent.agents.default import DefaultAgent


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


def build_summary(agent: "DefaultAgent", *, execution_time_seconds: float = 0.0) -> dict[str, Any]:
    """Build a human-readable summary dict from an agent's state.

    Args:
        agent: The agent instance (provides ``messages``, ``cost``, ``n_calls``).
        execution_time_seconds: Wall-clock seconds the run took.

    Returns:
        A JSON-serialisable summary dictionary.
    """
    messages = agent.messages
    cost = agent.cost
    llm_responses: list[dict] = []
    operations: list[dict] = []
    cell_edits: dict[int, list[str]] = {}  # cell_index -> [code_v1, code_v2, ...]

    last_assistant: dict = {}
    step = 0
    for idx, msg in enumerate(messages):
        role = msg.get("role", "")

        if role == "assistant":
            step += 1
            last_assistant = msg
            content = msg.get("content", "") or ""
            reasoning = extract_reasoning(msg)

            llm_responses.append({
                "step": step,
                "content": content,
                "reasoning": reasoning,
            })

            # Track edit_cell actions for code-change tracking
            for action in msg.get("extra", {}).get("actions", []):
                if action.get("tool_name") == "edit_cell":
                    args = action.get("arguments", {})
                    cell_index = int(args.get("cell_index", -1))
                    code = args.get("code", "")
                    if cell_index >= 0:
                        cell_edits.setdefault(cell_index, []).append(code)

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            extra = msg.get("extra", {})
            raw_output = extra.get("raw_output", "")
            returncode = extra.get("returncode", 0)

            # Find the matching command from the preceding assistant message
            matched = find_action(last_assistant.get("extra", {}).get("actions", []), tool_call_id)
            command = matched.get("command", "") if matched else ""

            operations.append({
                "step": step,
                "action": command,
                "output": raw_output,
                "return_code": returncode,
                "success": returncode == 0,
            })

        elif role == "exit":
            extra = msg.get("extra", {})
            exit_status = extra.get("exit_status", "")
            submission = extra.get("submission", "")
            operations.append({
                "step": step,
                "action": exit_status,
                "output": submission,
                "return_code": 0,
                "success": exit_status == "Submitted",
            })

    # Determine overall success
    success = any(
        op.get("action") == "Submitted" and op.get("success")
        for op in operations
    )

    # Build code changes list
    code_changes = []
    for cell_index in sorted(cell_edits.keys()):
        edits = cell_edits[cell_index]
        code_changes.append({
            "cell_id": cell_index,
            "fixed_code": edits[-1] if edits else "",
            "edit_count": len(edits),
        })

    successful_ops = sum(1 for op in operations if op.get("success", False))

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time_seconds, 2),
            "success": success,
            "status": "SUCCESS" if success else "INCOMPLETE",
            "cost": round(cost, 4),
        },
        "statistics": {
            "total_steps": len(llm_responses),
            "total_operations": len(operations),
            "successful_operations": successful_ops,
            "failed_operations": len(operations) - successful_ops,
            "cells_edited": len(cell_edits),
        },
        "llm_responses": llm_responses,
        "operations": operations,
        "code_changes": code_changes,
    }
