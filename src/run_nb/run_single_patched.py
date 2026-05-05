from __future__ import annotations

"""Independent patched-notebook execution flow.

This module is intentionally independent from the agent and NotebookEnvironment flow.
It provides a simple path to execute a generated *_patched.py notebook script.
"""

import shutil
import traceback
from pathlib import Path

import nbformat

from data_analysis.util_test_cells_evaluation import (
    VALIDATED_TEST_NOTEBOOK_EXCLUSIONS,
    _patched_script_to_cells,
)
from src.notebook_environment import NotebookEnvironmentConfig
from src.run_nb.docker_executor import DockerNotebookExecutor
from src.run_nb.setup_tmp_folder import cleanup_tmp_workdir, setup_tmp_workdir
from src.utils.log import logger


def _build_error_output_path(patched_script: Path) -> Path:
    return patched_script.with_name(f"{patched_script.stem}_error.txt")


def _validate_inputs(patched_script: Path, instance_folder: Path) -> None:
    if not patched_script.exists():
        raise FileNotFoundError(f"Patched script not found: {patched_script}")
    if not instance_folder.exists():
        raise FileNotFoundError(f"Instance folder not found: {instance_folder}")
    if not patched_script.is_file():
        raise FileNotFoundError(f"Patched script is not a file: {patched_script}")
    if not instance_folder.is_dir():
        raise FileNotFoundError(f"Instance folder is not a directory: {instance_folder}")


def _stage_inputs(patched_script: Path, instance_folder: Path, tmp_root: Path) -> tuple[Path, Path]:
    staged_instance_dir = tmp_root / instance_folder.name
    shutil.copytree(instance_folder, staged_instance_dir)

    staged_patched_script = staged_instance_dir / patched_script.name
    shutil.copy2(patched_script, staged_patched_script)
    return staged_instance_dir, staged_patched_script


def _load_last_benchmark_test_cell_source(instance_folder: Path) -> str | None:
    instance_name = instance_folder.name
    notebook_path = instance_folder / f"{instance_name}_fixed_with_test_executed.ipynb"

    if instance_name.lower() in VALIDATED_TEST_NOTEBOOK_EXCLUSIONS or not notebook_path.exists():
        return None

    notebook = nbformat.read(notebook_path, as_version=4)
    if not notebook.cells:
        return None

    source = notebook.cells[-1].get("source", "")
    if isinstance(source, list):
        source = "".join(source)

    if not isinstance(source, str) or not source.strip():
        return None
    return source


def _build_notebook_from_patched_script(staged_patched_script: Path, instance_folder: Path) -> Path:
    patch_source = staged_patched_script.read_text(encoding="utf-8")
    patch_cells = _patched_script_to_cells(patch_source)
    if not patch_cells:
        raise RuntimeError(f"No code cells parsed from patched script: {staged_patched_script}")

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source=source) for source in patch_cells]

    test_cell_source = _load_last_benchmark_test_cell_source(instance_folder)
    if test_cell_source:
        notebook.cells.append(nbformat.v4.new_code_cell(source=test_cell_source))

    notebook_path = staged_patched_script.with_suffix(".ipynb")
    nbformat.write(notebook, notebook_path)
    return notebook_path


def run_single_patched_notebook(
    patched_script: Path,
    instance_name: str,
    config: NotebookEnvironmentConfig | dict,
) -> Path | None:
    """Execute one generated *_patched.py via a staged notebook in Docker.

    Returns:
        Path to the copied executed notebook next to the patched script.
    """
    config = (
        config
        if isinstance(config, NotebookEnvironmentConfig)
        else NotebookEnvironmentConfig.model_validate(config)
    )
    tmp_root = Path(config.docker_mount_path).resolve() / f"patched_{config.port}"
    instance_folder = Path(config.source_path_parent) / instance_name
    _validate_inputs(patched_script, instance_folder)
    setup_tmp_workdir(tmp_root, docker_image_name=config.docker_image_name)
    output_notebook_path = patched_script.with_name(f"{patched_script.stem}_executed.ipynb")
    output_error_path = patched_script.with_name(f"{patched_script.stem}_error.txt")

    try:
        staged_instance_dir, staged_patched_script = _stage_inputs(patched_script, instance_folder, tmp_root)
        staged_notebook_path = _build_notebook_from_patched_script(staged_patched_script, staged_instance_dir)
        if output_error_path.exists():
            output_error_path.unlink()

        with DockerNotebookExecutor(
            mount_host_dir=tmp_root,
            docker_image_name=config.docker_image_name,
            timeout_seconds=config.timeout,
        ) as executor:
            try:
                exit_code, stdout, stderr = executor.execute(
                    mounted_folder=staged_instance_dir,
                    notebook_name=staged_notebook_path.name,
                )
            finally:
                executor.restore_mount_ownership()

        output_notebook_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_notebook_path, output_notebook_path)

        if exit_code != 0:
            raise RuntimeError(
                "Notebook execution failed\n"
                f"returncode={exit_code}\n\n"
                f"stdout:\n{stdout}\n\n"
                f"stderr:\n{stderr}"
            )

        return output_notebook_path
    except Exception as e:
        error_output_path = _build_error_output_path(patched_script)
        error_payload = (
            f"patched_path={patched_script}\n"
            f"instance_folder={instance_folder}\n"
            f"exception={e}\n\n"
            "traceback:\n"
            f"{traceback.format_exc()}"
        )
        error_output_path.write_text(error_payload, encoding="utf-8")
        logger.error(f"Error during patched notebook execution; wrote error file: {error_output_path}")
        return None
    finally:
        try:
            cleanup_tmp_workdir(tmp_root)
        except Exception as cleanup_exc:  # noqa: BLE001
            logger.error(f"Error during patched notebook cleanup: {cleanup_exc}")
