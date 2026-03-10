import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Optional, TypeAlias

import typer
from src.utils.log import logger
from src.utils.ui import get_progress_advance_fn, progress_live, set_ui_enabled
from src.utils.yaml_parser import (
    load_config,
    load_api_keys,
    set_model_config,
    get_run_count,
    get_instances,
    get_trajectories_dir,
    prepare_config_for_threading
)
from src.run_agent import run_single_instance, get_instance_summary_path
from src.run_baseline import run_baseline_instance

app = typer.Typer(rich_markup_mode="rich")
# --config points to the run-specific overlay. load_config() merges exactly two layers:
# defaults from NOTEBOOK_AGENT_DEFAULTS_PATH (or config/defaults.yaml) and this overlay.
CONFIG_PATH = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "./config/agent.yaml"))
DEFAULTS_PATH = Path(os.getenv("NOTEBOOK_AGENT_DEFAULTS_PATH", "./config/defaults.yaml"))

SingleInstance: TypeAlias = tuple[str, int, Path]
ProgressAdvanceFn: TypeAlias = Callable[[str], None]
RunNonThreadedFn: TypeAlias = Callable[[dict, SingleInstance, ProgressAdvanceFn], None]


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

def should_skip_instance(output_dir, instance_name) -> bool:
    """Check if an instance should be skipped based on summary status.
    
    Returns True if the instance should be skipped (already completed successfully),
    False if it should be run (doesn't exist or has failed/incomplete status).
    """
    summary_path = get_instance_summary_path(output_dir, instance_name)

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


def build_instances(config: dict, model_name: str) -> list[SingleInstance]:
    misc_config = config.get("misc", {})
    skip_existing = misc_config.get("skip_existing", False)
    trajectories_dir = get_trajectories_dir(config)
    number_of_runs = get_run_count(config)

    IGNORED_INSTANCES = {"torch_13"}
    IGNORED_PREFIXES = ("NBspecific",)

    runs_to_execute = []
    for instance_name in get_instances(config):
        if instance_name in IGNORED_INSTANCES or instance_name.startswith(IGNORED_PREFIXES):
            logger.info(f"Skipping (filtered): {instance_name}")
            continue
        for run_num in range(1, number_of_runs + 1):
            run_output_dir = trajectories_dir / model_name / f"run_{run_num}"
            if skip_existing:
                if should_skip_instance(run_output_dir, instance_name):
                    logger.info(f"Skipping (already completed): {instance_name} run {run_num}")
                    continue
            run_output_dir.mkdir(parents=True, exist_ok=True)
            runs_to_execute.append((instance_name, run_num, run_output_dir))
    
    return runs_to_execute


def get_run_threaded_fn(config: dict, api_keys: list[str], runs_to_execute: list[SingleInstance], run_non_threaded: RunNonThreadedFn) -> Callable[[ProgressAdvanceFn], None]:
    def run_threaded(progress_advance_fn: ProgressAdvanceFn):
        run_queue: Queue[SingleInstance] = Queue()

        for run_item in runs_to_execute:
            run_queue.put(run_item)

        def worker(config: dict, worker_index: int):
            logger.info(f"Worker {worker_index} starting.")
            while True:
                try:
                    run = run_queue.get_nowait()
                except Empty:
                    logger.info(f"Worker {worker_index} has no more runs to process and is exiting.")
                    break
                logger.info(f"Worker {worker_index} picked up run: {run[0]} run {run[1]}")
                run_non_threaded(config, run, progress_advance_fn)
                logger.debug(f"Worker {worker_index} finished run.")
                run_queue.task_done()

        with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
            futures = [
                executor.submit(worker, prepare_config_for_threading(config, api_key, index), index)
                for index, api_key in enumerate(api_keys)
            ]
            run_queue.join()
            for future in futures:
                future.result()
    return run_threaded

# fmt: off
@app.command()
def main(
    config_spec: Path = typer.Option(CONFIG_PATH, "-c", "--config", help="Path to run-specific config file"),
    model_name: Optional[str] = typer.Option(None, "-m", "--model", help="Model name to use (defaults to first model in config)"),
    enable_threading: bool = typer.Option(False, "--threads/--no-threads", help="Enable multi-threaded execution"),
    api_keys_file: Optional[Path] = typer.Option(None, "--api-keys", help="Path to file with API keys (one per line, required for threading)"),
) -> Any:

    # Load and parse layered configuration: defaults layer + run-specific overlay.
    config = load_config(config_spec, DEFAULTS_PATH)
    
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
        if not api_keys:
            logger.error(f"No API keys found in file: {api_keys_file}")
            return None
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
            
    # Build list of SingleInstance to execute
    instances_to_execute: list[SingleInstance] = build_instances(config, model_name)
    
    # Calculate total iterations for progress bar
    number_of_instances = len(instances_to_execute)
    
    if number_of_instances == 0:
        logger.info(f"No runs to execute for model {model_name} - all already completed")
        return None

    run_fn = run_baseline_instance if is_baseline else run_single_instance

    def run_non_threaded(config: dict, instance: SingleInstance, progress_advance_fn: ProgressAdvanceFn):
            instance_name, run_num, output_dir = instance
            
            try:
                run_fn(instance_name, config, output_dir)
                # Check if run completed successfully, and retry once if not
                if not should_skip_instance(output_dir, instance_name):
                    logger.warning(f"Run failed or incomplete for {instance_name} run {run_num}, retrying once...")
                    run_fn(instance_name, config, output_dir)
            except Exception as e:
                logger.error(f"Failed to run model={model_name}, instance={instance_name}, run={run_num}: {e}")
                logger.exception(e)
            finally:
                progress_advance_fn(f"{mode_label}: {model_name} | {instance_name} | run {run_num}")
    
    run_threaded = get_run_threaded_fn(config, api_keys, instances_to_execute, run_non_threaded) 

    with progress_live(number_of_instances, f"{mode_label}: {model_name}") as (progress, task_id):
        progress_advance_fn: ProgressAdvanceFn = get_progress_advance_fn(
            progress=progress,
            task_id=task_id,
            total_iterations=number_of_instances,
        )
        if enable_threading:
            run_threaded(progress_advance_fn)
        else:
            for instance in instances_to_execute:
                run_non_threaded(config, instance, progress_advance_fn=progress_advance_fn)

    # Summary: check which instances did not complete successfully
    incomplete_instances = list(filter(lambda i: not should_skip_instance(i[2], i[0]), instances_to_execute))
    
    if len(incomplete_instances) > 0:
        logger.warning(f"Failed to complete {len(incomplete_instances)} out of {len(instances_to_execute)} runs:")
        map(lambda i: logger.warning(f"  - {i[0]} (run {i[1]})"), incomplete_instances)
    else:
        logger.info(f"All {len(instances_to_execute)} runs completed successfully")


if __name__ == "__main__":
    app()

