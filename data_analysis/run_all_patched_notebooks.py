import os
import sys
from pathlib import Path
from typing import Any, Iterable
import typer

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Config CLI defaults (match main.py behavior)
CONFIG_PATH = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "config/agent.yaml"))
DEFAULTS_PATH = Path(os.getenv("NOTEBOOK_AGENT_DEFAULTS_PATH", "config/defaults.yaml"))

if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.util_test_cells_evaluation import (
    _extract_instance_name_from_filename,
)
from src.run_nb.run_single_patched import run_single_patched_notebook
from src.utils.yaml_parser import get_trajectories_dir, load_config


def _load_full_config(config_spec: Path = CONFIG_PATH) -> dict[str, Any]:
    _load_dotenv(PROJECT_ROOT / ".env")
    return load_config(config_spec, DEFAULTS_PATH)

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


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"" , "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)



app = typer.Typer()


@app.command()
def main(
    config: Path = typer.Option(CONFIG_PATH, "-c", "--config", help="Path to run-specific config file"),
):
    """Run patched notebook execution using the provided agent config."""    
    print(f"Loading configuration from: {config.resolve()}")
    full_config = _load_full_config(config)
    results_root = get_trajectories_dir(full_config)
    environment_config = full_config.get("environment", {})
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
