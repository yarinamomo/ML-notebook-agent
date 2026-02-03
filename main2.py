import mini_swe_agent_examples as run_mini_agent

import os
import traceback
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from minisweagent.config import  get_config_path
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.utils.save import save_traj
from src.notebook_environment import NotebookEnvironment
from src.LoggingLitellmModel import LoggingLitellmModel
from src.utils.log import logger


app = typer.Typer(rich_markup_mode="rich")
DEFAULT_CONFIG = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "./config/default.yaml"))
DEFAULT_OUTPUT = Path(os.getenv("NOTEBOOK_AGENT_TRAJECTORIES_PATH", "./trajectories/last_run.traj.json"))
console = Console(highlight=False)


# fmt: off
@app.command(help="_HELP_TEXT")
def main(
    model_name: str | None = typer.Option( None, "-m", "--model", help="Model to use",),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
) -> Any:
    config_path = get_config_path(config_spec)
    console.print(f"Loading agent config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text())
    console.print(f"Configuration loaded: {config}")
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if model_name is not None:
        config.setdefault("model", {})["model_name"] = model_name

    model = LoggingLitellmModel(**config.get("model", {}))
        
    target_nb_instance = "sklearn_1"
    source_path = f"example/JunoBench/{target_nb_instance}"

    env = NotebookEnvironment(
        sandbox_settings={
            "image_name": "yarinamomo/kaggle_python_env",
            "port": 8888,
        },
        source_path=source_path,
        docker_mount_path="example/docker_mount/",
        with_debugger=False,
        log_file="agent_interaction.log",  # Enable logging to file
        verbose=True,  # Set to False to disable console output
    )
    agent = DefaultAgent(model, env, **config.get("agent", {}))
    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run("Fix the bug in the notebook")  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
    return agent


if __name__ == "__main__":
    app()

