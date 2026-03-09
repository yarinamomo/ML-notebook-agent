"""Unit tests for NotebookEnvironment submit gating behavior."""

import pytest
from minisweagent.exceptions import Submitted

from src.notebook_environment import NotebookEnvironment


class FakeBenchmarkProblem:
    """Minimal benchmark stub to control notebook execution outcomes in tests."""

    def __init__(self, *args, **kwargs):
        self._cell_count = 1
        self._run_all_results = [
            {
                "execution_count": 1,
                "status": "ok",
                "done": True,
                "outputs": [],
            }
        ]
        self._run_cell_result = {
            "execution_count": 2,
            "status": "ok",
            "done": True,
            "outputs": [],
        }

    def get_cell_count(self):
        return self._cell_count

    def get_cells(self):
        return [
            {
                "cell_type": "code",
                "execution_count": None,
                "source": "print('hello')",
                "metadata": {},
                "outputs": [],
            }
        ]

    def get_cell(self, index):
        return {
            "cell_type": "code",
            "execution_count": None,
            "source": "print('hello')",
            "metadata": {},
            "outputs": [],
        }

    def edit_cell(self, index, code):
        return None

    def run_cell(self, index):
        return self._run_cell_result

    def run_all(self):
        return self._run_all_results

    def execute_python_command(self, code):
        return {
            "execution_count": 3,
            "status": "ok",
            "done": True,
            "outputs": [],
        }

    def close(self):
        return None


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr("src.notebook_environment.BenchmarkProblem", FakeBenchmarkProblem)
    return NotebookEnvironment(instance_name="dummy", output_dir=tmp_path)


def _submit_exit_status(env: NotebookEnvironment) -> str:
    with pytest.raises(Submitted) as exc_info:
        env.execute({"tool_name": "submit", "arguments": {"summary": "done"}})

    payload = exc_info.value.messages[0]
    return payload["extra"]["exit_status"]


def test_submit_is_invalid_before_successful_run_all(env):
    assert _submit_exit_status(env) == "SubmittedWithErrors"


def test_submit_is_valid_after_successful_run_all(env):
    run_result = env.execute({"tool_name": "run_all", "arguments": {}})

    assert run_result["returncode"] == 0
    assert _submit_exit_status(env) == "Submitted"


def test_submit_becomes_invalid_after_edit_following_run_all(env):
    env.execute({"tool_name": "run_all", "arguments": {}})
    edit_result = env.execute(
        {
            "tool_name": "edit_cell",
            "arguments": {"cell_index": 0, "code": "print(42)"},
        }
    )

    assert edit_result["returncode"] == 0
    assert _submit_exit_status(env) == "SubmittedWithErrors"


def test_submit_becomes_invalid_after_error_following_run_all(env):
    env.execute({"tool_name": "run_all", "arguments": {}})

    env.problem._run_cell_result = {
        "execution_count": 2,
        "status": "error",
        "done": True,
        "outputs": [
            {
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "boom",
                "traceback": ["ValueError: boom"],
            }
        ],
    }

    run_cell_result = env.execute({"tool_name": "run_cell", "arguments": {"cell_index": 0}})

    assert run_cell_result["returncode"] == 1
    assert _submit_exit_status(env) == "SubmittedWithErrors"


@pytest.mark.parametrize(
    "action",
    [
        {"tool_name": "get_cell_count", "arguments": {}},
        {"tool_name": "get_cells", "arguments": {}},
        {"tool_name": "get_cell", "arguments": {"cell_index": 0}},
        {"tool_name": "run_code", "arguments": {"code": "x = 1"}},
    ],
)
def test_submit_stays_valid_after_non_mutating_actions(action, env):
    env.execute({"tool_name": "run_all", "arguments": {}})

    action_result = env.execute(action)

    assert action_result["returncode"] == 0
    assert _submit_exit_status(env) == "Submitted"
