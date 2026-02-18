import os
import traceback
from pathlib import Path
from typing import Any

import typer
import yaml
from minisweagent.config import  get_config_path
from minisweagent.run.utils.save import save_traj
from src.notebook_environment import NotebookEnvironment
from src.LoggingLitellmModel import LoggingLitellmModel
from src.utils.log import logger
from src.ui_agent import UiAgent
from src.utils.summary_logger import initialize_logger, get_logger

app = typer.Typer(rich_markup_mode="rich")
DEFAULT_CONFIG = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "./config/default.yaml"))
DEFAULT_OUTPUT = Path(os.getenv("NOTEBOOK_AGENT_TRAJECTORIES_PATH", "./trajectories/last_run.traj.json"))


# fmt: off
@app.command(help="_HELP_TEXT")
def main(
    model_name: str | None = typer.Option( None, "-m", "--model", help="Model to use",),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
) -> Any:

    logger.info("Starting notebook agent....")
    config_path = get_config_path(config_spec)
    config = yaml.safe_load(config_path.read_text())
    logger.debug(f"Configuration loaded: {config}")

    misc_config = config.get("misc", {})
    if misc_config.get("if_local_key", False):
        logger.info("Loading API keys from .env file")
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)

    # Initialize summary logger    
    if misc_config.get("enable_summary_log", False):
        initialize_logger(
            enabled=True,
            output_path=misc_config.get("summary_log_path", "./trajectories/last_run_summary.json")
        )
        logger.info("Summary logging enabled")
        
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if model_name is not None:
        config.setdefault("model", {})["model_name"] = model_name

    model = LoggingLitellmModel(**config.get("model", {}))
    
    env_config = config.get("environment", {})
    target_nb_instance = env_config.get("target_nb_instance", "sklearn_1")

    env = NotebookEnvironment(
        sandbox_settings={
            "image_name": env_config.get("docker_image_name", "yarinamomo/junobench-simple"),
            "port": 8888,
            "start_command": env_config.get("docker_start_command", None)
        },
        source_path=env_config.get("source_path_parent", "example/JunoBench/")+target_nb_instance,
        docker_mount_path=env_config.get("docker_mount_path", "example/docker_mount/"),
        problem_mode=env_config.get("problem_mode", "JunoBench_Buggy"),
        timeout=env_config.get("timeout", 30)
    )
    agent = UiAgent(model, env, **config.get("agent", {}))
    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run("Fix the crashes in the notebook")  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Check if task completed successfully
        if misc_config.get("enable_summary_log", False):
            if result and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(result):
                get_logger().mark_success(step=agent.model.n_calls)
                logger.info("Task completed successfully")
            # Save summary and trajectory
            get_logger().save_summary()
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
        env.close()
    return agent


if __name__ == "__main__":
    app()

