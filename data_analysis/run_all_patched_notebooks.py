import sys
import traceback
from pathlib import Path


from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.util_test_cells_evaluation import (
    _extract_instance_name_from_filename,
)
from data_analysis.run_single_patched_notebook import run_single_patched_notebook

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_JUNO_BENCH_ROOT = Path("JunoBench")
DEFAULT_SETTING_ORDER = (
    "agent",
    "baseline",
    "agent_without_run_code_and_cell_outputs",
    "baseline_without_all_outputs",
    "baseline_without_cell_outputs",
)


def discover_patched_notebooks(results_root: Path, setting: str, model: str) -> list[Path]:
    model_root = results_root / setting / model
    if not model_root.exists():
        raise FileNotFoundError(f"Results folder not found: {model_root}")
    return sorted(model_root.glob("run_*/*_patched.py"))


def extract_instance_name_from_patched_path(patched_path: Path) -> str:
    stem = patched_path.stem
    if stem.endswith("_patched"):
        return stem[: -len("_patched")]
    return _extract_instance_name_from_filename(patched_path)


def discover_setting_model_pairs(results_root: Path) -> list[tuple[str, str]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    setting_dirs = [path for path in results_root.iterdir() if path.is_dir()]
    setting_names = {path.name for path in setting_dirs}

    ordered_settings = [name for name in DEFAULT_SETTING_ORDER if name in setting_names]
    ordered_settings.extend(sorted(setting_names - set(ordered_settings)))

    pairs: list[tuple[str, str]] = []
    for setting in ordered_settings:
        setting_dir = results_root / setting
        model_dirs = [path for path in setting_dir.iterdir() if path.is_dir()]
        for model_dir in sorted(model_dirs):
            if any(model_dir.glob("run_*/*_patched.py")):
                pairs.append((setting, model_dir.name))

    return pairs


def build_executed_output_path(patched_path: Path) -> Path:
    return patched_path.with_name(f"{patched_path.stem}_executed.ipynb")


def build_error_output_path(patched_path: Path) -> Path:
    return patched_path.with_name(f"{patched_path.stem}_error.txt")

def run_all_settings_with_defaults() -> None:
    results_root = DEFAULT_RESULTS_ROOT
    juno_bench_root = DEFAULT_JUNO_BENCH_ROOT
    benchmark_root = juno_bench_root / "benchmark"
    setting_model_pairs = discover_setting_model_pairs(results_root)
    if not setting_model_pairs:
        raise FileNotFoundError(f"No patched notebooks found under: {results_root}")

    for setting, model in setting_model_pairs:
        print(f"Starting setting/model: {setting}/{model}")
        patched_paths = discover_patched_notebooks(results_root, setting, model)
        for patched_path in tqdm(patched_paths, desc=f"Run patched [{setting}/{model}]", unit="notebook"):
            executed_output_path = build_executed_output_path(patched_path)
            error_output_path = build_error_output_path(patched_path)

            if executed_output_path.exists() and not error_output_path.exists():
                print(f"Skipping existing executed notebook: {executed_output_path}")
                continue

            if error_output_path.exists():
                print(f"Re-running due to existing error file: {error_output_path}")

            instance_name = extract_instance_name_from_patched_path(patched_path)
            instance_folder = benchmark_root / instance_name

            try:
                run_single_patched_notebook(
                    patched_script=patched_path,
                    instance_folder=instance_folder,
                )
                if error_output_path.exists():
                    error_output_path.unlink()
            except Exception as exc:  # noqa: BLE001
                error_payload = (
                    f"setting={setting}\n"
                    f"model={model}\n"
                    f"patched_path={patched_path}\n"
                    f"instance_folder={instance_folder}\n"
                    f"exception={exc}\n\n"
                    "traceback:\n"
                    f"{traceback.format_exc()}"
                )
                error_output_path.write_text(error_payload, encoding="utf-8")
                print(f"Failed {patched_path}; wrote error file: {error_output_path}")



if __name__ == "__main__":
    run_all_settings_with_defaults()
