from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import nbformat


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "data_analysis" / "run_patched_notebooks_in_docker.py"
    spec = importlib.util.spec_from_file_location("run_patched_notebooks_in_docker", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module()


def write_validated_test_notebook(path: Path, last_cell_source: str) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(source="setup = True"),
        nbformat.v4.new_code_cell(source=last_cell_source),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)


def write_patched_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#%%\nvalue = 1\n\n#%%\nprint(value)\n",
        encoding="utf-8",
    )


def read_notebook(path: Path) -> nbformat.NotebookNode:
    with path.open("r", encoding="utf-8") as handle:
        return nbformat.read(handle, as_version=4)


def test_build_execution_notebook_appends_last_test_cell(tmp_path: Path):
    juno_root = tmp_path / "JunoBench"
    benchmark_root = juno_root / "benchmark"
    instance_dir = benchmark_root / "sklearn_1"
    patched_path = tmp_path / "results" / "agent" / "model-a" / "run_1" / "sklearn_1_patched.py"

    write_validated_test_notebook(instance_dir / "sklearn_1_fixed_with_test_executed.ipynb", "assert value == 1")
    write_patched_script(patched_path)

    notebook = runner.build_execution_notebook(patched_path, benchmark_root)

    assert len(notebook.cells) == 3
    assert notebook.cells[0].source == "value = 1"
    assert notebook.cells[1].source == "print(value)"
    assert notebook.cells[2].source == "assert value == 1"


def test_process_single_patched_notebook_retries_recoverable_failure(tmp_path: Path, monkeypatch):
    results_root = tmp_path / "results"
    juno_root = tmp_path / "JunoBench"
    benchmark_root = juno_root / "benchmark"
    instance_dir = benchmark_root / "sklearn_1"
    patched_path = results_root / "agent" / "model-a" / "run_1" / "sklearn_1_patched.py"
    output_path = patched_path.with_name("sklearn_1_patched_executed.ipynb")

    write_validated_test_notebook(instance_dir / "sklearn_1_fixed_with_test_executed.ipynb", "assert value == 1")
    write_patched_script(patched_path)

    config = runner.RunnerConfig(
        setting="agent",
        model="model-a",
        results_root=results_root,
        juno_bench_root=juno_root,
        benchmark_root=benchmark_root,
        temp_root=juno_root / "tmp" / "patched_notebook_runs",
        container_mount_path="/app/container",
        docker_image_name="test-image",
        docker_start_command="jupyter server --allow-root --ServerApp.ip=0.0.0.0 --ServerApp.port=8888 --ServerApp.token='' --ServerApp.password='' --ServerApp.open_browser=False",
        port=runner.DEFAULT_PORT,
        notebook_timeout=30,
        max_retries=2,
        retry_delay_seconds=0,
        overwrite=True,
    )

    class FakeExecutor:
        call_count = 0

        def __init__(self, *args, **kwargs):
            self.container_name = "fake-container"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute_notebook(self, notebook_host_path: Path, timeout_seconds: int):
            self.__class__.call_count += 1
            if self.__class__.call_count == 1:
                raise runner.RecoverableNotebookExecutionError("container crashed")
            return runner.ExecutionAttemptResult(stdout="ok", stderr="", returncode=0, timed_out=False)

        def stop(self):
            return None

    monkeypatch.setattr(runner, "DockerNotebookExecutor", FakeExecutor)
    monkeypatch.setattr(runner.time, "sleep", lambda *_args, **_kwargs: None)

    record = runner.process_single_patched_notebook(patched_path, config)

    assert record.status == "completed"
    assert record.attempts == 2
    assert output_path.exists()
    executed_nb = read_notebook(output_path)
    assert executed_nb.cells[-1].source == "assert value == 1"


def test_process_single_patched_notebook_copies_artifact_on_nonrecoverable_failure(tmp_path: Path, monkeypatch):
    results_root = tmp_path / "results"
    juno_root = tmp_path / "JunoBench"
    benchmark_root = juno_root / "benchmark"
    instance_dir = benchmark_root / "sklearn_1"
    patched_path = results_root / "agent" / "model-a" / "run_1" / "sklearn_1_patched.py"
    output_path = patched_path.with_name("sklearn_1_patched_executed.ipynb")

    write_validated_test_notebook(instance_dir / "sklearn_1_fixed_with_test_executed.ipynb", "assert value == 1")
    write_patched_script(patched_path)

    config = runner.RunnerConfig(
        setting="agent",
        model="model-a",
        results_root=results_root,
        juno_bench_root=juno_root,
        benchmark_root=benchmark_root,
        temp_root=juno_root / "tmp" / "patched_notebook_runs",
        container_mount_path="/app/container",
        docker_image_name="test-image",
        docker_start_command="jupyter server --allow-root --ServerApp.ip=0.0.0.0 --ServerApp.port=8888 --ServerApp.token='' --ServerApp.password='' --ServerApp.open_browser=False",
        port=runner.DEFAULT_PORT,
        notebook_timeout=30,
        max_retries=1,
        retry_delay_seconds=0,
        overwrite=True,
    )

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            self.container_name = "fake-container"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute_notebook(self, notebook_host_path: Path, timeout_seconds: int):
            raise runner.NonRecoverableNotebookExecutionError("nbconvert failed with CellExecutionError")

        def stop(self):
            return None

    monkeypatch.setattr(runner, "DockerNotebookExecutor", FakeExecutor)

    record = runner.process_single_patched_notebook(patched_path, config)

    assert record.status == "notebook_failed"
    assert record.attempts == 1
    assert output_path.exists()
    executed_nb = read_notebook(output_path)
    assert executed_nb.cells[-1].source == "assert value == 1"
