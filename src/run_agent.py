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

def get_instance_summary_path(run_output_dir: Path, instance_name: str) -> Path:
    return run_output_dir / f"{instance_name}_summary.json"


def run_single_instance(
    instance_name: str,
    config: dict,
    output_dir: Path,
) -> tuple[str, Optional[str]]:
    """Run a single instance with a specific model and run number.
    
    Returns:
        tuple: (exit_status, submission)
    """
    # Create model instance
    model = CustomToolLitellmModel(**config.get("model", {}))
    
    # Create environment
    env = NotebookEnvironment(
        instance_name=instance_name,
        output_dir=output_dir,
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
        instance_traj_path = get_instance_trajectory_path(output_dir, instance_name)
        agent.save(instance_traj_path)
        logger.info(f"Saved trajectory to: {instance_traj_path}")
        summary_path = get_instance_summary_path(output_dir, instance_name)
        agent.save_summary(summary_path)
        logger.info(f"Saved summary to: {summary_path}")
        
        # Close environment
        env.close()
        time.sleep(2)  # Ensure clean shutdown
    
    return exit_status, submission
