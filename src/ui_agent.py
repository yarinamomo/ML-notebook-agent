import json
import time
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow
import src.utils.ui as ui
from src.utils.summary_util import build_summary, extract_reasoning, find_action
from typing import override, TYPE_CHECKING

if TYPE_CHECKING:
    from src.notebook_environment import NotebookEnvironment
class AgentTimeout(InterruptAgentFlow):
    """Raised when the agent exceeds its configured total timeout."""

class EnvironmentUnavailable(InterruptAgentFlow):
    """Raised when the environment is unavailable or encounters a critical error."""

class UiAgent(DefaultAgent):
    env: "NotebookEnvironment"

    def __init__(self, *args, **kwargs):
        self.total_timeout: int = kwargs.pop("total_timeout", 0)
        super().__init__(*args, **kwargs)
        self._start_time: float | None = None

    @override
    def run(self, task: str = "", **kwargs) -> dict:
        """ """
        self._start_time = time.monotonic()
        # Store initial cells before the agent starts modifying the notebook
        if hasattr(self.env, 'problem') and self.env.problem:
            self._initial_cells = self.env.problem.get_cells()
        else:
            self._initial_cells = []
        return super().run(task, **kwargs)

    @override
    def query(self) -> dict:
        if self.total_timeout > 0 and self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self.total_timeout:
                raise AgentTimeout({
                    "role": "exit",
                    "content": "AgentTimeout",
                    "extra": {"exit_status": "AgentTimeout", "submission": f"Agent execution exceeded total timeout of {self.total_timeout} seconds"},
                })

        with ui.wait_llm():
            response = super().query()
        actions = "\n".join([action.get("command", "") for action in response.get("extra", {}).get("actions", [])])
        reasoning = extract_reasoning(response)
        ui.system(self.n_calls, response.get("content", ""), actions=actions, reasoning=reasoning)
        return response

    @override
    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", []) # command, tool_call_id
        commands = "\n".join([action.get("command", "") for action in actions])
        with ui.wait_tool(commands):
            response_list = super().execute_actions(message)
            for response in response_list:
                tool_call_id = response.get('tool_call_id', '')
                matched = find_action(actions, tool_call_id)
                cmd = matched.get('command', '') if matched else None
                extra = response.get('extra', {})
                output = extra.get('raw_output', '')
                returncode = extra.get('returncode', '')
                # timestamp = extra.get('timestamp', '')
                # exception_info = extra.get('exception_info', '')
                ui.agent(self.n_calls, output, action=cmd, return_code=returncode)

        return response_list
    

    def save_summary(self, path: Path | None = None) -> dict:
        """Build an execution summary from the message history. Save to *path* if given."""
        elapsed = (time.monotonic() - self._start_time) if self._start_time else 0.0
        initial_cells = getattr(self, '_initial_cells', [])
        summary = build_summary(self, execution_time_seconds=elapsed, initial_cells=initial_cells)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary