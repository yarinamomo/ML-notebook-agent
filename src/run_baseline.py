"""
Baseline runner: single-shot notebook debugging.

Unlike the agentic approach (which loops with the model), this baseline:
1. Runs all cells once to collect errors.
2. Sends the full notebook + error outputs to the model in a single prompt.
3. The model responds with edit_cell calls to fix all errors at once.
4. Runs all cells again to verify the fixes.
"""

import json
import time
from pathlib import Path
from typing import Optional, cast

import litellm

from src.notebook_environment import NotebookEnvironment
from src.CustomToolLitellmModel import CustomToolLitellmModel
from src.notebook_tools import (
    EDIT_CELL_TOOL,
    parse_notebook_tool_actions,
    parse_tool_calls_from_content,
)
from src.utils.log import logger
from src.utils.summary_util import build_summary
from src.run_agent import get_instance_summary_path, get_instance_trajectory_path


# ---------------------------------------------------------------------------
# Baseline model: only exposes edit_cell to the LLM
# ---------------------------------------------------------------------------

BASELINE_TOOLS = [EDIT_CELL_TOOL]


class BaselineLitellmModel(CustomToolLitellmModel):
    """LLM model that only exposes edit_cell as available tool."""

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=BASELINE_TOOLS,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e


# ---------------------------------------------------------------------------
# Baseline execution
# ---------------------------------------------------------------------------

def run_baseline_instance(
    instance_name: str,
    config: dict,
    output_dir: Path,
    api_key: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Run the baseline (single-shot) approach on one instance.

    Returns:
        (exit_status, summary_text)
    """
    start_time = time.monotonic()

    model_config = config.get("model", {})
    if api_key:
        model_config = model_config.copy()
        model_config["api_key"] = api_key

    model = BaselineLitellmModel(**model_config)
    system_template = config.get("agent", {}).get("system_template", "")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- set up environment ---
    env = NotebookEnvironment(
        instance_name=instance_name,
        output_dir=output_dir,
        **config.get("environment", {}),
    )

    exit_status = ""
    summary_text: Optional[str] = None
    submission: Optional[str] = None
    initial_cells: list[str] = []
    
    # Track messages and cost for build_summary
    messages: list[dict] = []
    response_cost = None
    try:
        # 1) Read all cells
        initial_cells = env.problem.get_cells()
        cells_text = [env.problem.get_cell(i) for i in range(env.problem.get_cell_count())]

        # Add system and user messages
        messages.append(model.format_message(role="system", content=system_template))

        user_prompt = (
            "Here is the full notebook:\n\n"
            f"{chr(10).join(cells_text)}\n\n"
            "---\n\n"
        )

        # 2) Run all to get error output
        run_all_result = env.execute({
            "tool_name": "run_all",
            "arguments": {},
            "tool_call_id": "baseline_run_all_0",
            "command": "run_all()",
        })
        initial_run_output = run_all_result["output"]
        initial_returncode = run_all_result["returncode"]

        logger.info(
            f"[{instance_name}] Initial run_all returncode={initial_returncode}, "
            f"output length={len(initial_run_output)}"
        )

        # If no errors at all, nothing to fix
        if initial_returncode == 0:
            exit_status = "Submitted"
            summary_text = "No errors found in initial run — nothing to fix."
        else:
            # Complete the user message with error output
            user_prompt += (
                "Here is the output from running all cells:\n\n"
                f"{initial_run_output}\n\n"
                "---\n\n"
                "Analyze the errors above and fix ALL crashed cells using the edit_cell tool. "
                "You may call edit_cell multiple times (once per cell). "
                "Do NOT comment out code to avoid errors — keep the original intent."
            )
            messages.append(model.format_message(role="user", content=user_prompt))

            # 3) Query the model once
            response = model.query(messages)
            actions = response.get("extra", {}).get("actions", [])
            # Add assistant message
            messages.append(response)

            logger.info(
                f"[{instance_name}] Model returned {len(actions)} edit_cell action(s), cost: {response.get("extra", {}).get("cost", 'unknown')}"
            )

            # 4) Apply edits

            outputs = [cast(dict, env.execute(action)) for action in response.get("extra", {}).get("actions", [])]
            messages.extend(model.format_observation_messages(response, outputs))

            # 5) Run all again to verify
            verify_result = env.execute({
                "tool_name": "run_all",
                "arguments": {},
                "tool_call_id": "baseline_run_all_verify",
                "command": "run_all()",
            })
            exit_status = "Success" if verify_result["returncode"] == 0 else "Failure"
            submission = verify_result.get("output")
            summary_text = f"Applied {len(actions)} edit(s)"

    except Exception as e:
        exit_status = type(e).__name__
        summary_text = str(e)
        logger.error(f"Error in baseline run: {e}", exc_info=True)
    finally:
        elapsed = time.monotonic() - start_time

        # Add exit message
        messages.append({
            "role": "exit",
            "content": summary_text,
            "extra": {
                "exit_status": exit_status,
                "submission": submission
            },
        })
        
        trajectory_path = get_instance_trajectory_path(output_dir, instance_name)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory = {"messages": messages}
        trajectory_path.write_text(json.dumps(trajectory, indent=2))

        # Build summary using build_summary utility
        summary_path = get_instance_summary_path(output_dir, instance_name, config)
        if summary_path:
            summary = build_summary(messages, response_cost, elapsed, initial_cells)
            
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2))
            logger.info(f"Saved baseline summary to: {summary_path}")

        env.close()
        time.sleep(2)

    return exit_status, summary_text
