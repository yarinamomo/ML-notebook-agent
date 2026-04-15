from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import nbformat

LLM_NAME_MAPPING = {
    "glm-4.7-355b": "glm",
    # Add more mappings as needed
}

VALIDATED_TEST_NOTEBOOK_EXCLUSIONS = {
    "pandas_11",
    "statsmodels_1",
    "statsmodels_2",
    "seaborn_1",
    "seaborn_2",
    "seaborn_3",
    "seaborn_5",
    "seaborn_6",
    "matplotlib_2",
    "matplotlib_3",
    "matplotlib_4",
    "matplotlib_5",
    "tensorflow_9",
}

def convert_sampled_patches_to_notebooks(
    setting: str,
    model: str,
    output_dir: str | Path = "JunoBench/test_file_genartion/testpatchedfiles_baseline",
):
    """Convert sampled patched.py scripts to notebooks with one appended test cell.

    Output naming format:
    {setting}_{llm}_run_x_library_y.ipynb
    where llm is parsed as the model prefix before the first '-' (for example,
    model='glm-4.7-355b' -> llm='glm').
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sampled_patches = _extract_sampled_patched_files(setting=setting, model=model)
    llm_name = LLM_NAME_MAPPING[model]

    created_files: list[Path] = []
    for sampled_key, patch_path in sampled_patches.items():
        run_name, instance_name = sampled_key.split("/", 1)

        patch_source = patch_path.read_text(encoding="utf-8")
        patch_cells = _patched_script_to_cells(patch_source)
        if not patch_cells:
            raise ValueError(f"No code cells parsed from patched script: {patch_path}")

        notebook = nbformat.v4.new_notebook()
        notebook.cells = [nbformat.v4.new_code_cell(source=source) for source in patch_cells]

        test_last_cell_source = _load_last_test_cell_source(instance_name)
        if test_last_cell_source:
            notebook.cells.append(nbformat.v4.new_code_cell(source=test_last_cell_source))

        notebook_name = f"{setting}_{llm_name}_{run_name}_{instance_name}.ipynb"
        notebook_path = output_root / notebook_name
        nbformat.write(notebook, notebook_path)
        created_files.append(notebook_path)

    return created_files

def _extract_sampled_patched_files(
    setting: str,
    model: str
    ) -> dict[str, Path]:
    """Resolve sampled patched.py files for a given results setting/model.

    The sampled keys are read from:
    results/{setting}/{model}/analysis/stratified_sampled_instances_labeled.json

    Returns a mapping like:
    {
        "run_1/matplotlib_1": Path(".../results/.../run_1/matplotlib_1_patched.py"),
        ...
    }
    """
    results_base = Path("results") / setting / model
    sampled_json_path = results_base / "analysis" / "stratified_sampled_instances_labeled.json"

    if not sampled_json_path.exists():
        raise FileNotFoundError(f"Sampled instances JSON not found: {sampled_json_path}")

    sampled_data = json.loads(sampled_json_path.read_text(encoding="utf-8"))
    sampled_instances = sampled_data.get("sampled_instances", {})
    if not isinstance(sampled_instances, dict):
        raise ValueError("'sampled_instances' must be a dictionary in sampled JSON")

    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for sampled_key in sampled_instances:
        try:
            run_name, instance_name = sampled_key.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sampled instance key '{sampled_key}'. Expected format 'run_x/library_y'."
            ) from exc

        patch_path = results_base / run_name / f"{instance_name}_patched.py"
        if not patch_path.exists():
            missing.append(sampled_key)
            continue

        resolved[sampled_key] = patch_path

    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            "Could not find one or more sampled patched files: "
            f"{missing_list} under {results_base}"
        )

    return resolved

def _patched_script_to_cells(script_text: str) -> list[str]:
    """Split a patched.py file into notebook code-cell sources."""
    parts = re.split(r"^\s*#%%\s*$", script_text, flags=re.MULTILINE)
    cells = [part.strip("\n") for part in parts]
    return [cell for cell in cells if cell.strip()]


def _load_last_test_cell_source(instance_name: str) -> str:
    """Get the last cell source from the corresponding testfiles_validated notebook."""
    instance_dir = Path("JunoBench") / "test_file_genartion" / "testfiles_validated"
    if not instance_dir.exists():
        print(f"Testfiles instance folder not found: {instance_dir}")
        return ""
    notebook_path = _pick_test_notebook(instance_dir, instance_name)
    if not notebook_path:
        return ""
    notebook = nbformat.read(notebook_path, as_version=4)
    if not notebook.cells:
        raise ValueError(f"Notebook has no cells: {notebook_path}")

    last_cell = notebook.cells[-1]
    source = last_cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"Last cell source is empty in notebook: {notebook_path}")

    return source

def _pick_test_notebook(instance_dir: Path, instance_name: str) -> Path | None:
    """Select the test notebook for an instance, preferring executed fixed notebooks."""
    pattern = f"{instance_name}_fixed_with_test_executed.ipynb"

    candidates = sorted(instance_dir.glob(pattern))
    if candidates:
        return candidates[0]

    print(f"No notebook {instance_name} found under {instance_dir}")
    return None


def copy_notebooks_to_benchmark(
    source_dir: str | Path,
    benchmark_dir: str | Path = "JunoBench/benchmark",
) -> None:
    """Copy all source notebooks into benchmark instance folders by instance name."""
    source_root = Path(source_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    benchmark_root = Path(benchmark_dir)
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {benchmark_root}")

    copied_files: list[Path] = []

    if source_root.name == "testfiles":
        notebooks = sorted(source_root.glob("*/*.ipynb"))
    else:
        # Copy all notebooks (including nested folders) from non-testfiles sources.
        notebooks = sorted(source_root.rglob("*.ipynb"))

    for notebook_path in notebooks:
        if source_root.name == "testfiles":
            instance_name = notebook_path.parent.name
        else:
            instance_name = _extract_instance_name_from_filename(notebook_path)

        target_dir = benchmark_root / instance_name
        destination = target_dir / notebook_path.name
        shutil.copy2(notebook_path, destination)
        copied_files.append(destination)
    print(f"Copied {len(copied_files)} notebooks from {source_root} to {benchmark_root}")


def copy_validated_test_notebooks(
    benchmark_dir: str | Path = "JunoBench/benchmark",
    output_dir: str | Path = "JunoBench/test_file_genartion/testfiles_validated",
) -> list[Path]:
    """Copy executed fixed notebooks from benchmark into validated testfiles folders."""
    benchmark_root = Path(benchmark_dir)
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {benchmark_root}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []
    missing_notebooks: list[Path] = []

    for instance_dir in sorted(benchmark_root.iterdir()):
        if not instance_dir.is_dir() or instance_dir.name.startswith("."):
            continue

        instance_name = instance_dir.name
        if instance_name.lower() in VALIDATED_TEST_NOTEBOOK_EXCLUSIONS:
            continue

        if instance_name.lower().startswith("nbspecific"):
            continue

        source_notebook = instance_dir / f"{instance_name}_fixed_with_test_executed.ipynb"
        if not source_notebook.exists():
            missing_notebooks.append(source_notebook)
            continue

        destination = output_root / source_notebook.name
        shutil.copy2(source_notebook, destination)
        copied_files.append(destination)

    if missing_notebooks:
        missing_list = "\n".join(str(path) for path in missing_notebooks)
        raise FileNotFoundError(
            "Could not find one or more validated notebooks to copy:\n"
            f"{missing_list}"
        )

    print(f"Copied {len(copied_files)} validated notebooks from {benchmark_root} to {output_root}")
    return copied_files

def _extract_instance_name_from_filename(notebook_path: Path) -> str:
    """Infer instance name (library_x) from a notebook filename."""
    stem = notebook_path.stem

    generated_match = re.search(r"_run_\d+_(.+_\d+)$", stem)
    if generated_match:
        return generated_match.group(1)

    fixed_match = re.match(r"(.+_\d+)_fixed(?:_with_test_executed)?$", stem)
    if fixed_match:
        return fixed_match.group(1)

    generic_match = re.match(r"(.+_\d+)$", stem)
    if generic_match:
        return generic_match.group(1)

    raise ValueError(f"Cannot infer instance name from notebook filename: {notebook_path.name}")


def _label_token(value: Any) -> str:
    """Normalize a label value into a stable token for marginal counting."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(value)


def _calculate_cohen_kappa_from_labels(labels1: list[Any], labels2: list[Any]) -> float:
    """Calculate Cohen's kappa from two aligned label lists."""
    if len(labels1) != len(labels2):
        raise ValueError("labels1 and labels2 must have the same length")

    total = len(labels1)
    if total == 0:
        return 0.0

    agreements = sum(1 for v1, v2 in zip(labels1, labels2) if v1 == v2)
    p_o = agreements / total

    tokens1 = [_label_token(value) for value in labels1]
    tokens2 = [_label_token(value) for value in labels2]
    counts1 = Counter(tokens1)
    counts2 = Counter(tokens2)
    all_tokens = set(counts1) | set(counts2)

    p_e = sum((counts1[token] / total) * (counts2[token] / total) for token in all_tokens)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


def _disagreement_sort_key(lib_name: str) -> tuple[str, int]:
    lib_match = re.match(r"(.+?)_(\d+)$", lib_name)
    if lib_match:
        lib_prefix = lib_match.group(1)
        lib_idx = int(lib_match.group(2))
    else:
        lib_prefix = lib_name
        lib_idx = 10**9

    return (lib_prefix, lib_idx)


def _build_pair_details(sampled1: dict[str, Any], sampled2: dict[str, Any]) -> dict[str, Any]:
    """Build aligned labels and disagreement details for one pair."""
    all_keys = sorted(set(sampled1.keys()) | set(sampled2.keys()))
    labels1: list[Any] = []
    labels2: list[Any] = []
    disagreement_keys: list[str] = []

    for key in all_keys:
        val1 = sampled1.get(key)
        val2 = sampled2.get(key)
        labels1.append(val1)
        labels2.append(val2)
        if val1 != val2:
            disagreement_keys.append(key)

    disagreement_libs = {
        key.partition("/")[2] if "/" in key else key for key in disagreement_keys
    }

    return {
        "all_keys": all_keys,
        "labels1": labels1,
        "labels2": labels2,
        "disagreement_keys": disagreement_keys,
        "disagreement_libs": sorted(disagreement_libs, key=_disagreement_sort_key),
    }


def calculate_kappa_score(path1: Path, path2: Path) -> float:
    # Load the JSON files
    with open(path1, encoding="utf-8") as f:
        data1 = json.load(f)
    with open(path2, encoding="utf-8") as f:
        data2 = json.load(f)

    # Extract sampled instances dictionaries
    sampled1 = data1.get("sampled_instances", {})
    sampled2 = data2.get("sampled_instances", {})

    pair_details = _build_pair_details(sampled1, sampled2)
    total = len(pair_details["all_keys"])
    if total == 0:
        return 0.0

    kappa = _calculate_cohen_kappa_from_labels(pair_details["labels1"], pair_details["labels2"])
    details = "\n".join(pair_details["disagreement_libs"])
    print(
        f"Kappa score: {kappa}, \n"
        f"Disagreements: {len(pair_details['disagreement_keys'])}/{total}.\n"
        f"Disagreed tests details:\n{details}"
    )
    return kappa


def calculate_overall_kappa_score(pairs: list[tuple[Path, Path]]) -> dict[str, Any]:
    """Pool multiple pairs by concatenating aligned labels and compute one overall kappa."""
    if not pairs:
        return {
            "overall_score": 0.0,
            "per_pair_scores": [],
            "per_pair_disagreement_instances": [],
        }

    pooled_labels1: list[Any] = []
    pooled_labels2: list[Any] = []
    per_pair_scores: list[float] = []
    per_pair_disagreement_instances: list[dict[str, Any]] = []

    for index, (path1, path2) in enumerate(pairs, start=1):
        with open(path1, encoding="utf-8") as f:
            data1 = json.load(f)
        with open(path2, encoding="utf-8") as f:
            data2 = json.load(f)

        sampled1 = data1.get("sampled_instances", {})
        sampled2 = data2.get("sampled_instances", {})
        pair_details = _build_pair_details(sampled1, sampled2)

        pair_kappa = _calculate_cohen_kappa_from_labels(
            pair_details["labels1"],
            pair_details["labels2"],
        )
        per_pair_scores.append(pair_kappa)
        per_pair_disagreement_instances.append(
            {
                "pair_index": index,
                "path1": str(path1),
                "path2": str(path2),
                "score": pair_kappa,
                "disagreement_instances": pair_details["disagreement_libs"],
            }
        )

        pooled_labels1.extend(pair_details["labels1"])
        pooled_labels2.extend(pair_details["labels2"])

        details = "\n".join(pair_details["disagreement_libs"])
        print(
            f"Pair {index} kappa score: {pair_kappa}, \n"
            f"Disagreements: {len(pair_details['disagreement_keys'])}/{len(pair_details['all_keys'])}.\n"
            f"Disagreed tests details:\n{details}"
        )

    overall_score = _calculate_cohen_kappa_from_labels(pooled_labels1, pooled_labels2)
    print(f"Overall pooled kappa score: {overall_score}")
    return {
        "overall_score": overall_score,
        "per_pair_scores": per_pair_scores,
        "per_pair_disagreement_instances": per_pair_disagreement_instances,
    }




if __name__ == "__main__":
    # copy_validated_test_notebooks()

    # convert_sampled_patches_to_notebooks(setting="baseline", model="glm-4.7-355b")
    # # copy_notebooks_to_benchmark(source_dir="JunoBench/test_file_genartion/testfiles")
    # copy_notebooks_to_benchmark(source_dir="JunoBench/test_file_genartion/testpatchedfiles_baseline")

    calculate_overall_kappa_score(
        pairs=[
            (
                Path("results/agent/glm-4.7-355b/analysis/stratified_sampled_instances_labeled.json"),
                Path("results/agent/glm-4.7-355b/analysis/stratified_sampled_instances_tests.json"),
            ),
            (
                Path("results/agent_without_run_code_and_cell_outputs/glm-4.7-355b/analysis/stratified_sampled_instances_labeled.json"),
                Path("results/agent_without_run_code_and_cell_outputs/glm-4.7-355b/analysis/stratified_sampled_instances_tests.json"),
            ),
            (
                Path("results/baseline/glm-4.7-355b/analysis/stratified_sampled_instances_labeled.json"),
                Path("results/baseline/glm-4.7-355b/analysis/stratified_sampled_instances_tests.json"),
            ),
        ]
    )