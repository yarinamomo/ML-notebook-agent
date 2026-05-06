"""Unit tests for CustomToolLitellmModel tool selection."""

from types import SimpleNamespace

import litellm

from src.custom_tool_litellm_model import CustomToolLitellmModel
from src.notebook_tools import NOTEBOOK_TOOLS, NOTEBOOK_TOOLS_WITHOUT_RUN_CODE


def test_custom_tool_model_defaults_to_enabling_run_code():
    model = CustomToolLitellmModel(model_name="test-model")

    assert model.config.disable_run_code is False


def test_custom_tool_model_uses_run_code_tools_by_default(monkeypatch):
    captured = {}

    def fake_completion(*, model, messages, tools, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["tools"] = tools
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    model = CustomToolLitellmModel(model_name="test-model")
    model._query([{"role": "user", "content": "hello"}])

    assert captured["model"] == "test-model"
    assert captured["tools"] == NOTEBOOK_TOOLS


def test_custom_tool_model_can_disable_run_code(monkeypatch):
    captured = {}

    def fake_completion(*, model, messages, tools, **kwargs):
        captured["tools"] = tools
        return SimpleNamespace()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    model = CustomToolLitellmModel(model_name="test-model", disable_run_code=True)
    model._query([{"role": "user", "content": "hello"}])

    assert model.config.disable_run_code is True
    assert captured["tools"] == NOTEBOOK_TOOLS_WITHOUT_RUN_CODE