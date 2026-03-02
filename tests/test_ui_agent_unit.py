"""Unit tests for UiAgent error handling around sandbox and notebook environment failures."""

from contextlib import nullcontext
import time

import pytest

from src.ui_agent import AgentTimeout, EnvironmentUnavailable, UiAgent


class StubModel:
    def __init__(self, responses=None):
        self._responses = list(responses or [])

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
        return [
            {
                "role": "user",
                "content": outputs[0]["output"],
                "extra": {"outputs": outputs},
            }
        ]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class StubEnv:
    def __init__(self, execute_result=None, execute_exception=None):
        self._execute_result = execute_result or {
            "output": "ok",
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


@pytest.fixture
def quiet_ui(monkeypatch):
    monkeypatch.setattr("src.ui_agent.ui.wait_llm", lambda: nullcontext())
    monkeypatch.setattr("src.ui_agent.ui.wait_tool", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr("src.ui_agent.ui.system", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui_agent.ui.agent", lambda *_args, **_kwargs: None)


def test_query_raises_agent_timeout_before_model_call(quiet_ui):
    model = StubModel()
    env = StubEnv()
    agent = UiAgent(
        model=model,
        env=env,
        system_template="system",
        instance_template="instance",
        total_timeout=1,
    )
    agent._start_time = time.monotonic() - 5

    with pytest.raises(AgentTimeout) as exc_info:
        agent.query()

    timeout_message = exc_info.value.messages[0]
    assert timeout_message["content"] == "AgentTimeout"
    assert timeout_message["extra"]["exit_status"] == "AgentTimeout"


def test_run_stops_with_environment_unavailable_interrupt(quiet_ui):
    model = StubModel(
        responses=[
            {
                "role": "assistant",
                "content": "running",
                "extra": {
                    "actions": [
                        {
                            "command": "run_cell(0)",
                            "tool_name": "run_cell",
                            "arguments": {"cell_index": 0},
                            "tool_call_id": "tool-1",
                        }
                    ]
                },
            }
        ]
    )
    env = StubEnv(
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

    result = agent.run(task="reproduce notebook")

    assert result["exit_status"] == "EnvironmentUnavailable"
    assert "Kernel execution failed repeatedly" in result["submission"]


def test_execute_actions_surfaces_notebook_environment_error_output(quiet_ui):
    model = StubModel()
    env = StubEnv(
        execute_result={
            "output": "Error executing run_cell: RuntimeError('Kernel execution failed')",
            "returncode": 1,
            "exception_info": "Kernel execution failed",
        }
    )
    agent = UiAgent(
        model=model,
        env=env,
        system_template="system",
        instance_template="instance",
    )

    response_messages = agent.execute_actions(
        {
            "role": "assistant",
            "content": "run cell",
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
    )

    assert len(response_messages) == 1
    assert "Error executing run_cell" in response_messages[0]["content"]
    assert response_messages[0]["extra"]["outputs"][0]["returncode"] == 1
