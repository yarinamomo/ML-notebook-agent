"""
YAML configuration parser for notebook agent.
Handles configuration loading, parsing, and CLI overrides.
"""
import yaml
from pathlib import Path
from typing import Any
from minisweagent.config import get_config_path
from src.utils.log import logger


def load_config(config_spec: Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_spec: Path to config file
        
    Returns:
        Parsed configuration dictionary
    """
    config_path = get_config_path(config_spec)
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.debug(f"Configuration loaded from {config_path}: {config}")
    return config


def load_api_keys(config: dict[str, Any]) -> None:
    """
    Load API keys from .env file if configured.
    
    Args:
        config: Configuration dictionary
    """
    misc_config = config.get("misc", {})
    if misc_config.get("if_local_key", False):
        logger.info("Loading API keys from .env file")
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)


def apply_cli_overrides(
    config: dict[str, Any],
    cost_limit: float | None = None,
    run_count: int | None = None,
) -> dict[str, Any]:
    """
    Apply command-line overrides to configuration.
    
    Args:
        config: Configuration dictionary
        cost_limit: Override cost limit
        run_count: Override run count
        
    Returns:
        Updated configuration dictionary
    """
    # Apply cost limit override
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
        logger.info(f"Cost limit overridden to: {cost_limit}")
    
    # Store run_count override in misc section
    if run_count is not None:
        config.setdefault("misc", {})["run_count"] = run_count
        logger.info(f"Run count overridden to: {run_count}")
    
    return config


def set_model_config(config: dict[str, Any], model_name: str | None = None) -> str | None:
    """
    Select a model from the config and set it as config['model'].
    
    Args:
        config: Configuration dictionary (modified in place)
        model_name: Model name to select. If None, the first model is used.
        
    Returns:
        The selected model name, or None if selection failed
    """
    models_config = config.get("models", [])
    
    # If no models list, fall back to legacy single model config
    if not models_config:
        legacy_model = config.get("model", {})
        if not legacy_model:
            logger.error("No model configuration found in config")
            return None
        models_config = [legacy_model]
        logger.info("Using legacy single model configuration")
    
    if model_name:
        matching = [m for m in models_config if m.get("model_name") == model_name]
        if not matching:
            available = [m.get("model_name", "unknown") for m in models_config]
            logger.error(f"Model '{model_name}' not found in config. Available: {available}")
            return None
        model_config = matching[0]
    else:
        model_config = models_config[0]
    
    config["model"] = model_config
    selected_name = model_config.get("model_name", None)
    logger.info(f"Using model: {selected_name}")
    return selected_name


def get_run_count(config: dict[str, Any]) -> int:
    """
    Get the number of runs to execute.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Number of runs
    """
    misc_config = config.get("misc", {})
    run_count = misc_config.get("run_count", 1)
    logger.info(f"Run count: {run_count}")
    return run_count


def discover_instances(source_path_parent: Path) -> list[str]:
    """
    Discover all instance directories in the source_path_parent.
    
    Args:
        source_path_parent: Parent directory containing instances
        
    Returns:
        Sorted list of instance names
    """
    instances = []
    if not source_path_parent.exists():
        logger.warning(f"Source path parent does not exist: {source_path_parent}")
        return instances
    
    for item in source_path_parent.iterdir():
        if item.is_dir():
            instances.append(item.name)
    
    logger.info(f"Discovered {len(instances)} instances: {instances}")
    return sorted(instances)


def get_instances(config: dict[str, Any], run_all_override: bool | None = None) -> list[str]:
    """
    Get list of instances to run.
    
    Args:
        config: Configuration dictionary
        run_all_override: CLI override for run_all_instances flag
        
    Returns:
        List of instance names
    """
    env_config = config.get("environment", {})
    run_all_instances = (
        run_all_override if run_all_override is not None 
        else env_config.get("run_all_instances", False)
    )
    
    if run_all_instances:
        source_path_parent = Path(env_config.get("source_path_parent", "example/JunoBench/"))
        instances = discover_instances(source_path_parent)
        if not instances:
            logger.error(f"No instances found in {source_path_parent}")
            return []
    else:
        instances = [env_config.get("target_nb_instance", "sklearn_1")]
        logger.info(f"Running single instance: {instances[0]}")
    
    return instances


def get_trajectories_dir(config: dict[str, Any]) -> Path:
    """
    Get trajectories directory path from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to trajectories directory
    """
    misc_config = config.get("misc", {})
    trajectories_dir = Path(misc_config.get("trajectory_log_path", "./trajectories"))
    trajectories_dir.mkdir(exist_ok=True)
    return trajectories_dir

def apply_port_offset(config: dict[str, Any], port_offset: int) -> dict[str, Any]:
    if port_offset > 0:
        port = config.get("environment", {}).get("port", 8888) + port_offset
        print("Applying port offset:", port_offset, "-> New port:", port)
        config.get("environment",{})["port"] = port
    return config