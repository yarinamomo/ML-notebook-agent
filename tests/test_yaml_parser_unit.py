"""Unit tests for YAML parser helpers."""

from src.utils.yaml_parser import set_model_config


def test_set_model_config_selects_first_model_and_merges_defaults():
    config = {
        "model": {
            "temperature": 0.2,
            "model_kwargs": {
                "timeout": 30,
                "shared": True,
            },
        },
        "models": [
            {
                "model_name": "gpt-4o-mini",
                "model_kwargs": {
                    "api_key": "abc123",
                },
            },
            {
                "model_name": "gpt-4.1",
            },
        ],
    }

    selected_name = set_model_config(config)

    assert selected_name == "gpt-4o-mini"
    assert config["model"]["model_name"] == "gpt-4o-mini"
    assert config["model"]["temperature"] == 0.2
    assert config["model"]["model_kwargs"] == {
        "timeout": 30,
        "shared": True,
        "api_key": "abc123",
    }


def test_set_model_config_selects_named_model():
    config = {
        "model": {"temperature": 0.5},
        "models": [
            {"model_name": "small", "temperature": 0.1},
            {"model_name": "large", "temperature": 0.9},
        ],
    }

    selected_name = set_model_config(config, model_name="large")

    assert selected_name == "large"
    assert config["model"] == {"model_name": "large", "temperature": 0.9}


def test_set_model_config_returns_none_when_named_model_missing():
    config = {
        "models": [
            {"model_name": "small"},
        ],
    }

    selected_name = set_model_config(config, model_name="missing")

    assert selected_name is None
    assert config["models"] == [{"model_name": "small"}]
    assert "model" not in config


def test_set_model_config_falls_back_to_legacy_model_config():
    config = {
        "model": {
            "model_name": "legacy-model",
            "temperature": 0.4,
            "model_kwargs": {"timeout": 60},
        }
    }

    selected_name = set_model_config(config)

    assert selected_name == "legacy-model"
    assert config["model"] == {
        "model_name": "legacy-model",
        "temperature": 0.4,
        "model_kwargs": {"timeout": 60},
    }