"""Unit tests for UiAgent summary generation with summary_util integration."""

from contextlib import nullcontext
import json

import pytest

from src.ui_agent import EnvironmentUnavailable, UiAgent
from src.utils.summary_util import extract_reasoning, find_action


class _ProblemStub:
    def __init__(self, cells=None):
        self._cells = list(cells or ["print('original')"])

    def get_cells(self):
        return list(self._cells)


class _SummaryEnvStub:
    def __init__(self, *, execute_result=None, execute_exception=None, initial_cells=None):
        self.problem = _ProblemStub(initial_cells)
        self._execute_result = execute_result or {
            "output": "Cell 0 edited successfully",
            "returncode": 0,
            "exception_info": "",
        }
        self._execute_exception = execute_exception

    def execute(self, action, cwd=""):
        if self._execute_exception:
            raise self._execute_exception
        return self._execute_result

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class _SummaryModelStub:
    def __init__(self, responses):
        self._responses = list(responses)

    def query(self, messages, **kwargs):
        if not self._responses:
            return {
                "role": "exit",
                "content": "done",
                "extra": {"exit_status": "Submitted", "submission": ""},
            }
        return self._responses.pop(0)

    def format_message(self, **kwargs):
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        actions = message.get("extra", {}).get("actions", [])
        observations = []
        for action, output in zip(actions, outputs):
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": action.get("tool_call_id", ""),
                    "content": output["output"],
                    "extra": {
                        "raw_output": output["output"],
                        "returncode": output["returncode"],
                        "timestamp": 2.0,
                    },
                }
            )
        return observations

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


@pytest.fixture
def quiet_ui(monkeypatch):
    monkeypatch.setattr("src.ui_agent.ui.wait_llm", lambda: nullcontext())
    monkeypatch.setattr("src.ui_agent.ui.wait_tool", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr("src.ui_agent.ui.system", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui_agent.ui.agent", lambda *_args, **_kwargs: None)


def test_save_summary_includes_operations_code_changes_and_original_notebook(quiet_ui, tmp_path):
    model = _SummaryModelStub(
        responses=[
            {
                "role": "assistant",
                "content": "I will patch cell 0",
                "extra": {
                    "timestamp": 1.0,
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "reasoning_content": "Cell 0 likely has the failing line."
                                }
                            }
                        ]
                    },
                    "actions": [
                        {
                            "command": "edit_cell(0, 'print(42)')",
                            "tool_name": "edit_cell",
                            "arguments": {"cell_index": 0, "code": "print(42)"},
                            "tool_call_id": "tool-1",
                        }
                    ],
                },
            },
            {
                "role": "exit",
                "content": "done",
                "extra": {"exit_status": "Submitted", "submission": "All good"},
            },
        ]
    )
    env = _SummaryEnvStub(initial_cells=["print('before')"])
    agent = UiAgent(
        model=model,
        env=env,
        system_template="system",
        instance_template="instance",
    )

    run_result = agent.run(task="fix notebook")
    assert run_result["exit_status"] == "Submitted"

    output_path = tmp_path / "summary.json"
    summary = agent.save_summary(output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == summary

    assert summary["metadata"]["status"] == "Submitted"
    assert summary["metadata"]["success"] is True
    assert summary["original_notebook"] == ["print('before')"]
    assert summary["statistics"]["total_steps"] == 1
    assert summary["statistics"]["cells_edited"] == 1
    assert summary["statistics"]["unique_cells_edited"] == 1
    assert summary["statistics"]["total_operations"] == 2
    assert summary["operations"][0]["action"] == "edit_cell(0, 'print(42)')"
    assert summary["operations"][0]["success"] is True
    assert summary["operations"][1]["action"] == "Submitted"
    assert summary["code_changes"][0]["cell_index"] == 0
    assert summary["llm_responses"][0]["reasoning"] == "Cell 0 likely has the failing line."


def test_save_summary_marks_environment_unavailable_as_failed(quiet_ui):
    model = _SummaryModelStub(
        responses=[
            {
                "role": "assistant",
                "content": "running a cell",
                "extra": {
                    "actions": [
                        {
                            "command": "run_cell(1)",
                            "tool_name": "run_cell",
                            "arguments": {"cell_index": 1},
                            "tool_call_id": "tool-2",
                        }
                    ]
                },
            }
        ]
    )
    env = _SummaryEnvStub(
        execute_exception=EnvironmentUnavailable(
            {
                "role": "exit",
                "content": "EnvironmentUnavailable",
                "extra": {
                    "exit_status": "EnvironmentUnavailable",
                    "submission": "Kernel execution failed repeatedly after all retry attempts",
                },
            }
        )
    )
    agent = UiAgent(
        model=model,
        env=env,
        system_template="system",
        instance_template="instance",
    )

    run_result = agent.run(task="debug failure")
    assert run_result["exit_status"] == "EnvironmentUnavailable"

    summary = agent.save_summary()

    assert summary["metadata"]["status"] == "EnvironmentUnavailable"
    assert summary["metadata"]["success"] is False
    assert summary["statistics"]["total_steps"] == 1
    assert summary["statistics"]["total_operations"] == 1
    assert summary["operations"][0]["action"] == "EnvironmentUnavailable"
    assert summary["operations"][0]["success"] is False


def test_summary_util_helpers_extract_reasoning_and_find_action():
    message = {
        "extra": {
            "response": {
                "choices": [
                    {"message": {"reasoning_content": "fallback reasoning text"}}
                ]
            }
        }
    }
    assert extract_reasoning(message) == "fallback reasoning text"

    actions = [
        {"tool_call_id": "a1", "command": "run_cell(0)"},
        {"tool_call_id": "a2", "command": "submit()"},
    ]
    assert find_action(actions, "a2") == {"tool_call_id": "a2", "command": "submit()"}
    assert find_action(actions, "missing") is None
