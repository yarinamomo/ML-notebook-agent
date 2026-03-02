import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional
import threading

import typer
from src.notebook_environment import NotebookEnvironment, NotebookEnvironmentConfig
from src.CustomToolLitellmModel import CustomToolLitellmModel
from src.utils.log import logger
from src.ui_agent import UiAgent
from src.utils.ui import format_eta, progress_live
from src.utils.yaml_parser import (
    load_config,
    load_api_keys,
    apply_cli_overrides,
    get_models_config,
    get_run_count,
    get_instances,
    get_trajectories_dir
)

app = typer.Typer(rich_markup_mode="rich")
DEFAULT_CONFIG = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "./config/default.yaml"))


def run_single_instance(
    model_config: dict,
    instance_name: str,
    run_number: int,
    config: dict,
    trajectories_dir: Path
) -> tuple[str, Optional[str]]:
    """Run a single instance with a specific model and run number.
    
    Returns:
        tuple: (exit_status, submission)
    """
    
    model_name = model_config.get("model_name", "unknown")
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: Model={model_name}, Instance={instance_name}, Run={run_number}")
    logger.info(f"{'='*80}\n")
    
    # Create model instance
    model = CustomToolLitellmModel(**model_config)
    
    # Setup environment
    env_config = config.get("environment", {})
    source_path_parent = Path(env_config.get("source_path_parent", "example/JunoBench/"))
    
    # Create structured log paths: trajectories_dir/model_name/run_X/instance_name...
    run_output_dir = trajectories_dir / model_name / f"run_{run_number}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    misc_config = config.get("misc", {})

    instance_traj_path = run_output_dir / f"{instance_name}.traj.json"
    instance_summary_path = run_output_dir / f"{instance_name}_summary.json" if misc_config.get("enable_summary_log", False) else None
    
    # Skip if both trajectory and summary already exist
    if misc_config.get("skip_existing", False):
        if instance_traj_path.exists() and (instance_summary_path is None or instance_summary_path.exists()):
            logger.info(f"Skipping (already completed): {instance_name} run {run_number} — "
                        f"trajectory and summary already exist at {run_output_dir}")
            return "skipped", None

    
    # Create environment
    env = NotebookEnvironment(
        source_path=str(source_path_parent / instance_name),
        output_dir=str(run_output_dir),
        **env_config
    )

    # Create and run agent
    agent = UiAgent(model, env, **config.get("agent", {}))
    exit_status, submission = "", None
    
    # Get total_timeout from environment config
    total_timeout = env_config.get("total_timeout", 0)
    
    # Run agent with optional total timeout
    def run_agent():
        nonlocal exit_status, submission
        try:
            exit_info = agent.run("")  # type: ignore[arg-type]
            exit_status = exit_info.get("exit_status", "")
            submission = exit_info.get("submission", "")
        except Exception as e:
            exit_status, submission = type(e).__name__, str(e)
            raise
    
    try:
        if total_timeout is not None and total_timeout > 0:
            # Run with timeout
            thread = threading.Thread(target=run_agent)
            thread.daemon = True
            thread.start()
            thread.join(timeout=total_timeout)
            
            if thread.is_alive():
                # Timeout occurred
                logger.error(f"Agent run exceeded total timeout of {total_timeout}s")
                exit_status = "TIMEOUT"
                submission = f"Agent execution exceeded total timeout of {total_timeout} seconds"
        else:
            # Run without timeout
            run_agent()
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
    finally:
        # Save trajectory and summary
        agent.save(instance_traj_path)
        agent.save_summary(instance_summary_path)
        logger.info(f"Saved trajectory to: {instance_traj_path}")
        
        # Close environment
        env.close()
        import time
        time.sleep(2)  # Ensure clean shutdown
    
    return exit_status, submission

# fmt: off
@app.command(help="_HELP_TEXT")
def main(
    model_name: str | None = typer.Option( None, "-m", "--model", help="Model to use (overrides config)",),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    run_count: int | None = typer.Option(None, "-r", "--run-count", help="Number of runs (overrides config)"),
    run_all: bool | None = typer.Option(None, "--run-all/--single", help="Run all instances (overrides config)")
) -> Any:

    logger.info("Starting notebook agent....")
    
    # Load and parse configuration
    config = load_config(config_spec)
    load_api_keys(config)
    config = apply_cli_overrides(config, cost_limit, run_count)
    
    # Get configuration parameters
    models_config = get_models_config(config, model_name)
    total_run_count = get_run_count(config)
    instances = get_instances(config, run_all)
    trajectories_dir = get_trajectories_dir(config)
    
    # Validate instances
    if not instances:
        logger.error("No instances to run")
        return None
    
    # Main execution loops: models -> instances -> runs
    for model_config in models_config:
        model_name_str = model_config.get("model_name", "unknown")
        logger.info(f"\n{'#'*80}")
        logger.info(f"# Starting runs for model: {model_name_str}")
        logger.info(f"{'#'*80}\n")
        
        # Calculate total iterations for progress bar
        total_iterations = len(instances) * total_run_count

        with progress_live(total_iterations, f"Model: {model_name_str}") as (progress, task_id):

            start_time = time.monotonic()
            completed_count = 0

            for instance_name in instances:
                for run_num in range(1, total_run_count + 1):
                    try:
                        run_single_instance(
                            model_config=model_config,
                            instance_name=instance_name,
                            run_number=run_num,
                            config=config,
                            trajectories_dir=trajectories_dir
                        )
                        
                    except Exception as e:
                        logger.error(f"Failed to run model={model_name_str}, instance={instance_name}, run={run_num}: {e}")
                        logger.exception(e)
                    finally:
                        completed_count += 1
                        elapsed = time.monotonic() - start_time
                        avg_per_run = elapsed / completed_count
                        remaining = avg_per_run * (total_iterations - completed_count)
                        progress.update(
                            task_id,
                            advance=1,
                            description=f"Model: {model_name_str} | {instance_name} | run {run_num}",
                            eta=format_eta(remaining),
                        )

if __name__ == "__main__":
    app()

