import json
import time
from pathlib import Path
from typing import Optional

from src.notebook_environment import NotebookEnvironment
from src.CustomToolLitellmModel import CustomToolLitellmModel
from src.utils.log import logger
from src.ui_agent import UiAgent


def get_instance_trajectory_path(run_output_dir: Path, instance_name: str) -> Path:
    """Construct the trajectory file path for a given instance."""
    return run_output_dir / f"{instance_name}.traj.json"

def get_instance_summary_path(run_output_dir: Path, instance_name: str, config: dict) -> Optional[Path]:
    """Construct the summary file path for a given instance if summary logging is enabled."""
    enable_summary_log = config.get("misc", {}).get("enable_summary_log", False)
    if enable_summary_log:
        return run_output_dir / f"{instance_name}_summary.json"
    return None


def run_single_instance(
    instance_name: str,
    config: dict,
    run_output_dir: Path,
    api_key: Optional[str] = None
) -> tuple[str, Optional[str]]:
    """Run a single instance with a specific model and run number.
    
    Returns:
        tuple: (exit_status, submission)
    """
    
    # Override API key if provided (for threading)
    model_config = config.get("model", {})
    if api_key:
        model_config = model_config.copy()
        model_config["api_key"] = api_key
    
    # Create model instance
    model = CustomToolLitellmModel(**model_config)
    
    # Create environment
    env = NotebookEnvironment(
        instance_name=instance_name,
        output_dir=run_output_dir,
        **config.get("environment", {})
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
        instance_traj_path = get_instance_trajectory_path(run_output_dir, instance_name)
        agent.save(instance_traj_path)
        logger.info(f"Saved trajectory to: {instance_traj_path}")
        summary_path = get_instance_summary_path(run_output_dir, instance_name, config)
        agent.save_summary(summary_path)
        logger.info(f"Saved summary to: {summary_path}")
        
        # Close environment
        env.close()
        time.sleep(2)  # Ensure clean shutdown
    
    return exit_status, submission
