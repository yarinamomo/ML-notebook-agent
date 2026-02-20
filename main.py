import os
import traceback
from pathlib import Path
from typing import Any
import threading

import typer
from minisweagent.run.utils.save import save_traj
from src.notebook_environment import NotebookEnvironment
from src.LoggingLitellmModel import LoggingLitellmModel
from src.utils.log import logger
from src.ui_agent import UiAgent
from src.utils.summary_logger import (
    initialize_logger,
    get_logger,
    get_results_tracker
)
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
) -> tuple[str, Any, dict | None, float, int]:
    """Run a single instance with a specific model and run number.
    
    Returns:
        tuple: (exit_status, result, extra_info, cost, total_steps)
    """
    
    model_name = model_config.get("model_name", "unknown")
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: Model={model_name}, Instance={instance_name}, Run={run_number}")
    logger.info(f"{'='*80}\n")
    
    # Create model instance
    model = LoggingLitellmModel(**model_config)
    
    # Setup environment
    env_config = config.get("environment", {})
    source_path_parent = Path(env_config.get("source_path_parent", "example/JunoBench/"))
    
    # Create structured log paths: trajectories_dir/model_name/run_X/instance_name...
    run_output_dir = trajectories_dir / model_name / f"run_{run_number}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    instance_traj_path = run_output_dir / f"{instance_name}.traj.json"
    instance_summary_path = run_output_dir / f"{instance_name}_summary.json"
    
    # Initialize summary logger with instance-specific path
    misc_config = config.get("misc", {})
    if misc_config.get("enable_summary_log", False):
        initialize_logger(
            enabled=True,
            output_path=str(instance_summary_path)
        )
        logger.info(f"Summary logging enabled: {instance_summary_path}")
    
    # Create environment
    env = NotebookEnvironment(
        sandbox_settings={
            "image_name": env_config.get("docker_image_name", "yarinamomo/junobench-simple"),
            "port": 8888,
            "start_command": env_config.get("docker_start_command", None)
        },
        source_path=str(source_path_parent / instance_name),
        docker_mount_path=env_config.get("docker_mount_path", "example/docker_mount/"),
        problem_mode=env_config.get("problem_mode", "JunoBench_Buggy"),
        timeout=env_config.get("timeout", 30),
        run_all_timeout=env_config.get("run_all_timeout", 0),
        output_dir=str(run_output_dir)
    )
    
    # Create and run agent
    agent = UiAgent(model, env, **config.get("agent", {}))
    exit_status, result, extra_info, cost, total_steps = None, None, None, 0.0, 0
    
    # Get total_timeout from environment config
    total_timeout = env_config.get("total_timeout", 0)
    
    # Run agent with optional total timeout
    def run_agent():
        nonlocal exit_status, result
        try:
            exit_status, result = agent.run("")  # type: ignore[arg-type]
        except Exception as e:
            exit_status, result = type(e).__name__, str(e)
            raise
    
    try:
        if total_timeout and total_timeout > 0:
            # Run with timeout
            thread = threading.Thread(target=run_agent)
            thread.daemon = True
            thread.start()
            thread.join(timeout=total_timeout)
            
            if thread.is_alive():
                # Timeout occurred
                logger.error(f"Agent run exceeded total timeout of {total_timeout}s")
                exit_status = "TIMEOUT"
                result = f"Agent execution exceeded total timeout of {total_timeout} seconds"
                extra_info = {"reason": "total_timeout_exceeded"}
        else:
            # Run without timeout
            run_agent()
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
    finally:
        # Capture cost information
        cost = getattr(agent.model, 'cost', 0.0)
        total_steps = getattr(agent.model, 'n_calls', 0)
        
        # Check if task completed successfully
        if misc_config.get("enable_summary_log", False):
            if result and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(result):
                get_logger().mark_success(step=agent.model.n_calls)
                logger.info(f"Task completed successfully: {instance_name}")
            # Set cost and save summary
            get_logger().set_cost(cost)
            get_logger().save_summary()
        
        # Save trajectory
        save_traj(agent, instance_traj_path, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
        logger.info(f"Saved trajectory to: {instance_traj_path}")
        
        # Close environment
        env.close()
    
    return exit_status, result, extra_info, cost, total_steps

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
    
    # Get results tracker
    results_tracker = get_results_tracker()
    
    # Main execution loops: models -> instances -> runs
    for model_config in models_config:
        model_name_str = model_config.get("model_name", "unknown")
        logger.info(f"\n{'#'*80}")
        logger.info(f"# Starting runs for model: {model_name_str}")
        logger.info(f"{'#'*80}\n")
        
        for instance_name in instances:
            for run_num in range(1, total_run_count + 1):
                try:
                    exit_status, result, _, cost, total_steps = run_single_instance(
                        model_config=model_config,
                        instance_name=instance_name,
                        run_number=run_num,
                        config=config,
                        trajectories_dir=trajectories_dir
                    )
                    
                    success = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(result) if result else False
                    results_tracker.add_result(
                        model=model_name_str,
                        instance=instance_name,
                        run=run_num,
                        exit_status=exit_status,
                        success=success,
                        cost=cost,
                        total_steps=total_steps
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to run model={model_name_str}, instance={instance_name}, run={run_num}: {e}")
                    results_tracker.add_result(
                        model=model_name_str,
                        instance=instance_name,
                        run=run_num,
                        exit_status="FAILED",
                        success=False,
                        cost=0.0,
                        total_steps=0,
                        error=str(e)
                    )
    
    # Print and save execution summary
    results_tracker.print_summary()
    results_tracker.save_summary(trajectories_dir / "overall_summary.json")

if __name__ == "__main__":
    app()

