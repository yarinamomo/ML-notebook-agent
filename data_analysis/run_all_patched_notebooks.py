import os
import sys
from pathlib import Path
from typing import Any, Iterable
import typer

from tqdm import tqdm
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from the project root so module-level env reads respect overrides
load_dotenv(str(PROJECT_ROOT / ".env"), override=False)

# Config CLI defaults (match main.py behavior)
CONFIG_PATH = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "config/agent.yaml"))
DEFAULTS_PATH = Path(os.getenv("NOTEBOOK_AGENT_DEFAULTS_PATH", "config/defaults.yaml"))

app = typer.Typer()

if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.util_test_cells_evaluation import (
    _extract_instance_name_from_filename,
)
from src.run_nb.run_single_patched import run_single_patched_notebook
from src.utils.yaml_parser import get_trajectories_dir, load_config


def _load_full_config(config_spec: Path | None = None) -> dict[str, Any]:
    """Load the layered config using the already-loaded .env values.

    `.env` is loaded at module import time above, so environment overrides
    are available to resolve `CONFIG_PATH` and `DEFAULTS_PATH`.
    """
    resolved_config = config_spec if config_spec is not None else CONFIG_PATH
    return load_config(resolved_config, DEFAULTS_PATH)

def discover_patched_notebooks(
    model_root: Path,
    source_path_parent: Path,
) -> list[Path]:
    # `model_root` is a Path pointing to results_root/<setting>/<model>
    if not model_root.exists():
        raise FileNotFoundError(f"Results folder not found: {model_root}")
    discovered = model_root.glob("run_*/*_patched.py")
    filtered: list[Path] = []
    for patched_path in discovered:
        instance_name = extract_instance_name_from_patched_path(patched_path)
        if (source_path_parent / instance_name).is_dir():
            filtered.append(patched_path)

    return filtered


def extract_instance_name_from_patched_path(patched_path: Path) -> str:
    stem = patched_path.stem
    if stem.endswith("_patched"):
        return stem[: -len("_patched")]
    return _extract_instance_name_from_filename(patched_path)


def get_models(results_root: Path) -> list[Path]:
    """Return model directory Paths under results_root that contain patched scripts.

    Each returned Path points to results_root/<setting>/<model>.
    Preserves discovery order from glob traversal and avoids sorting.
    """
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    model_dirs: list[Path] = []
    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue
        # Expect structure: results_root/<model>/run_*/...*_patched.py
        if any(model_dir.glob("run_*/*_patched.py")):
            model_dirs.append(model_dir)

    return model_dirs


def build_executed_output_path(patched_path: Path) -> Path:
    return patched_path.with_name(f"{patched_path.stem}_executed.ipynb")


def build_error_output_path(patched_path: Path) -> Path:
    return patched_path.with_name(f"{patched_path.stem}_error.txt")



@app.command()
def main(
    config_spec: Path = typer.Option(CONFIG_PATH, "-c", "--config", help="Path to run-specific config file"),
):
    """Run patched notebook execution using the provided agent config."""    
    print(f"Loading configuration from: {config_spec.resolve()}")
    print(f"Using defaults from: {DEFAULTS_PATH.resolve()}")
    
    config = load_config(config_spec, DEFAULTS_PATH)
    results_root = get_trajectories_dir(config)
    environment_config = config.get("environment", {})
    source_path_parent = Path(environment_config.get("source_path_parent", "example/JunoBench/"))

    models = get_models(results_root)

    for  model in models:
        print(f"Starting model: /{model}")
        patched_paths = discover_patched_notebooks(
            model,
            source_path_parent,
        )
        for patched_path in tqdm(patched_paths, desc=f"Run patched [{model}]", unit="notebook"):
            executed_output_path = build_executed_output_path(patched_path)
            error_output_path = build_error_output_path(patched_path)

            if executed_output_path.exists() and not error_output_path.exists():
                print(f"Skipping existing executed notebook: {executed_output_path}")
                continue

            if error_output_path.exists():
                print(f"Re-running due to existing error file: {error_output_path}")

            instance_name = extract_instance_name_from_patched_path(patched_path)

            run_single_patched_notebook(
                patched_script=patched_path,
                instance_name=instance_name,
                config=environment_config,
            )
            if error_output_path.exists():
                print(f"Failed {patched_path}; error file: {error_output_path}")
            else:
                print(f"Completed {patched_path}")


if __name__ == "__main__":
    app()
