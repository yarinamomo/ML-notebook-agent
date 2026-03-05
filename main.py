import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from src.utils.log import logger
from src.utils.ui import get_progress_advance_fn, progress_live, set_ui_enabled
from src.utils.yaml_parser import (
    load_config,
    load_api_keys,
    set_model_config,
    get_run_count,
    get_instances,
    get_trajectories_dir
)
from src.run_agent import run_single_instance
from src.run_baseline import run_baseline_instance

app = typer.Typer(rich_markup_mode="rich")
DEFAULT_CONFIG = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "./config/default.yaml"))


def read_api_keys(api_keys_file: Path) -> list[str]:
    """Read API keys from a text file, one per line.
    
    Args:
        api_keys_file: Path to file containing API keys (one per line)
        
    Returns:
        List of API key strings
    """
    with open(api_keys_file, 'r') as f:
        keys = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return keys


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

# fmt: off
@app.command()
def main(
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    model_name: Optional[str] = typer.Option(None, "-m", "--model", help="Model name to use (defaults to first model in config)"),
    enable_threading: bool = typer.Option(False, "--threads/--no-threads", help="Enable multi-threaded execution"),
    api_keys_file: Optional[Path] = typer.Option(None, "--api-keys", help="Path to file with API keys (one per line, required for threading)"),
) -> Any:

    # Load and parse configuration
    config = load_config(config_spec)
    
    is_baseline = config.get("misc", {}).get("mode", "agent") == "baseline"
    mode_label = "Baseline" if is_baseline else "Agent"
    logger.info(f"Starting notebook {mode_label.lower()} run....")
    
    # Handle threading setup
    api_keys = []
    if enable_threading:
        if not api_keys_file:
            logger.error("--api-keys file is required when threading is enabled")
            return None
        if not api_keys_file.exists():
            logger.error(f"API keys file not found: {api_keys_file}")
            return None
        
        api_keys = read_api_keys(api_keys_file)
        logger.info(f"Threading enabled with {len(api_keys)} worker threads")
        
        # Disable UI for agent execution (worker threads)
        # But progress bar in main thread will still work
        set_ui_enabled(False)
        logger.info("Agent UI disabled for threaded execution (progress bar will still display)")
    else:
        load_api_keys(config)


    # Get configuration parameters
    model_name = set_model_config(config, model_name)
    if not model_name:
        logger.error("No valid model configuration found. Exiting.")
        return None
    
    total_run_count = get_run_count(config)
    instances = get_instances(config)
    trajectories_dir = get_trajectories_dir(config)
    
    # Validate instances
    if not instances:
        logger.error("No instances to run")
        return None
    
    # Build list of (instance, run) pairs to execute
    runs_to_execute = build_runs_to_execute(
        instances=instances,
        total_run_count=total_run_count,
        model_name=model_name,
        trajectories_dir=trajectories_dir,
        config=config
    )
    
    # Calculate total iterations for progress bar
    total_iterations = len(runs_to_execute)
    
    if total_iterations == 0:
        logger.info(f"No runs to execute for model {model_name} - all already completed")
        return None

    run_fn = run_baseline_instance if is_baseline else run_single_instance

    with progress_live(total_iterations, f"{mode_label}: {model_name}") as (progress, task_id):
        progress_advance_fn: Callable[[str], None] = get_progress_advance_fn(
            progress=progress,
            task_id=task_id,
            total_iterations=total_iterations,
        )

        for instance_name, run_num in runs_to_execute:
            try:
                run_output_dir = trajectories_dir / model_name / f"run_{run_num}"
                run_output_dir.mkdir(parents=True, exist_ok=True)

                run_fn(
                    instance_name=instance_name,
                    config=config,
                    run_output_dir=run_output_dir,
                )
                
            except Exception as e:
                logger.error(f"Failed to run model={model_name}, instance={instance_name}, run={run_num}: {e}")
                logger.exception(e)
            finally:
                progress_advance_fn(f"{mode_label}: {model_name} | {instance_name} | run {run_num}")

if __name__ == "__main__":
    app()

