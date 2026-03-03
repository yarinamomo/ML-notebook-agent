import os
import time
import traceback
import json
from pathlib import Path
from typing import Any, Optional

import typer
from src.notebook_environment import NotebookEnvironment, NotebookEnvironmentConfig
from src.CustomToolLitellmModel import CustomToolLitellmModel
from src.utils.log import logger
from src.ui_agent import UiAgent
from src.utils.ui import format_eta, progress_live, set_ui_enabled
from src.utils.yaml_parser import (
    load_config,
    load_api_keys,
    get_models_config,
    get_run_count,
    get_instances,
    get_trajectories_dir
)

app = typer.Typer(rich_markup_mode="rich")
DEFAULT_CONFIG = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "./config/default.yaml"))


def should_skip_instance(summary_path: Path) -> bool:
    """Check if an instance should be skipped based on summary status.
    
    Returns True if the instance should be skipped (already completed successfully),
    False if it should be run (doesn't exist or has failed/incomplete status).
    """
    if not summary_path.exists():
        return False
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        status = summary.get("metadata", {}).get("status", "")
        
        # Do NOT skip if status is EnvironmentUnavailable or INCOMPLETE
        if status in ["EnvironmentUnavailable", "INCOMPLETE"]:
            return False
        
        # Skip if status indicates successful completion
        return True
    except Exception as e:
        logger.warning(f"Failed to read summary {summary_path}: {e}")
        # If we can't read the summary, don't skip (safer to re-run)
        return False


def build_runs_to_execute(
    instances: list[str],
    total_run_count: int,
    model_name: str,
    trajectories_dir: Path,
    config: dict
) -> list[tuple[str, int]]:
    """Build a filtered list of (instance_name, run_number) pairs to execute.
    
    Filters out already completed runs based on summary status if skip_existing is enabled.
    
    Args:
        instances: List of instance names to process
        total_run_count: Number of runs per instance
        model_name: Name of the model being used
        trajectories_dir: Base directory for trajectory outputs
        config: Configuration dictionary
        
    Returns:
        List of (instance_name, run_number) tuples to execute
    """
    misc_config = config.get("misc", {})
    skip_existing = misc_config.get("skip_existing", False)
    enable_summary_log = misc_config.get("enable_summary_log", False)
    
    runs_to_execute = []
    for instance_name in instances:
        for run_num in range(1, total_run_count + 1):
            if skip_existing and enable_summary_log:
                run_output_dir = trajectories_dir / model_name / f"run_{run_num}"
                summary_path = run_output_dir / f"{instance_name}_summary.json"
                
                if should_skip_instance(summary_path):
                    logger.info(f"Skipping (already completed): {instance_name} run {run_num}")
                    continue
            
            runs_to_execute.append((instance_name, run_num))
    
    return runs_to_execute


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
    
    # Create environment
    env = NotebookEnvironment(
        source_path=str(source_path_parent / instance_name),
        output_dir=str(run_output_dir),
        **env_config
    )

    # Create and run agent
    agent = UiAgent(model, env, **config.get("agent", {}))
    exit_status, submission = "", None

    try:
        exit_info = agent.run("")  # type: ignore[arg-type]
        exit_status = exit_info.get("exit_status", "")
        submission = exit_info.get("submission", "")
    except Exception as e:
        exit_status, submission = type(e).__name__, str(e)
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
@app.command()
def main(
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
) -> Any:

    logger.info("Starting notebook agent....")
    
    # Load and parse configuration
    config = load_config(config_spec)
    load_api_keys(config)
    
    # Get configuration parameters
    models_config = get_models_config(config)
    total_run_count = get_run_count(config)
    instances = get_instances(config)
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
        
        # Build list of (instance, run) pairs to execute
        runs_to_execute = build_runs_to_execute(
            instances=instances,
            total_run_count=total_run_count,
            model_name=model_name_str,
            trajectories_dir=trajectories_dir,
            config=config
        )
        
        # Calculate total iterations for progress bar
        total_iterations = len(runs_to_execute)
        
        if total_iterations == 0:
            logger.info(f"No runs to execute for model {model_name_str} - all already completed")
            continue

        with progress_live(total_iterations, f"Model: {model_name_str}") as (progress, task_id):

            start_time = time.monotonic()
            completed_count = 0

            for instance_name, run_num in runs_to_execute:
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

