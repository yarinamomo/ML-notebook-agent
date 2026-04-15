from __future__ import annotations

import json
import os
import sys
import shutil
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import docker
import nbformat
import requests
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.util_test_cells_evaluation import (
    _extract_instance_name_from_filename,
    _patched_script_to_cells,
)
from src.utils.yaml_parser import load_config

DEFAULT_CONFIG_PATH = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "./config/agent.yaml"))
DEFAULT_DEFAULTS_PATH = Path(os.getenv("NOTEBOOK_AGENT_DEFAULTS_PATH", "./config/defaults.yaml"))
DEFAULT_TMP_ROOT = Path(os.getenv("NOTEBOOK_AGENT_TMP_PATH", str(PROJECT_ROOT / "tmp" / "patched_notebook_runs")))
DEFAULT_RETRY_DELAY_SECONDS = 5.0
DEFAULT_CONTAINER_MOUNT_PATH = "/app/container"
DEFAULT_PORT = 8888
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_JUNO_BENCH_ROOT = Path("JunoBench")
DEFAULT_MAX_RETRIES = 3
DEFAULT_SETTING_ORDER = (
    "agent",
    "baseline",
    "agent_without_run_code_and_cell_outputs",
    "baseline_without_all_outputs",
    "baseline_without_cell_outputs",
)


class NotebookAssemblyError(RuntimeError):
    """Raised when the patched script or test cell cannot be assembled."""


class RecoverableNotebookExecutionError(RuntimeError):
    """Raised when a notebook run should be retried with a fresh container."""


class NonRecoverableNotebookExecutionError(RuntimeError):
    """Raised when the notebook run failed in a way that should not be retried."""


@dataclass(slots=True)
class NotebookRunRecord:
    setting: str
    model: str
    run_name: str
    instance_name: str
    patched_path: str
    prepared_notebook_path: str
    executed_notebook_path: str
    output_path: str
    status: str
    attempts: int
    message: str
    container_name: str | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    test_cell_error: bool = False
    only_test_cell_error: bool = False
    execution_proof: bool = False
    execution_proof_detail: str = ""


@dataclass(slots=True)
class RunnerConfig:
    setting: str
    model: str
    results_root: Path
    juno_bench_root: Path
    benchmark_root: Path
    temp_root: Path
    container_mount_path: str
    docker_image_name: str
    docker_start_command: str | None
    port: int
    notebook_timeout: int
    max_retries: int
    retry_delay_seconds: float
    overwrite: bool


@dataclass(slots=True)
class ExecutionAttemptResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class DockerNotebookExecutor:
    def __init__(
        self,
        image_name: str,
        start_command: str | None,
        mount_host_root: Path,
        mount_container_path: str,
        port: int,
        container_name: str | None = None,
    ) -> None:
        self.image_name = image_name
        self.start_command = start_command
        self.mount_host_root = mount_host_root.resolve()
        self.mount_container_path = mount_container_path.rstrip("/")
        self.port = port
        self.container_name = container_name or f"patched-notebook-runner-{uuid.uuid4().hex[:8]}"
        self.client = docker.from_env()
        self.container = None

    def __enter__(self) -> "DockerNotebookExecutor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self._remove_existing_container()
        volumes = {
            str(self.mount_host_root): {
                "bind": self.mount_container_path,
                "mode": "rw",
            }
        }
        docker_kwargs: dict[str, Any] = {
            "name": self.container_name,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "volumes": volumes,
            "ports": {f"{self.port}/tcp": self.port},
            "command": self.start_command,
            "environment": {"HOME": "/app/container"},
        }

        if os.name == "posix":
            docker_kwargs["user"] = f"{os.getuid()}:{os.getgid()}"

        self.container = self.client.containers.run(self.image_name, **docker_kwargs)
        self._wait_for_server()

    def _remove_existing_container(self) -> None:
        try:
            old_container = self.client.containers.get(self.container_name)
        except Exception:
            return

        try:
            old_container.remove(force=True)
        except Exception:
            pass

    def _wait_for_server(self, timeout_seconds: int = 30) -> None:
        url = f"http://127.0.0.1:{self.port}/api/status"
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None

        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return
                last_error = RuntimeError(f"Jupyter status returned {response.status_code}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1)

        raise RecoverableNotebookExecutionError(
            f"Jupyter server did not become ready within {timeout_seconds} seconds"
        ) from last_error

    def stop(self) -> None:
        if self.container is None:
            return

        try:
            self.container.kill()
        except Exception:
            pass

        try:
            self.container.remove(force=True)
        except Exception:
            pass
        finally:
            self.container = None

    def container_path_for_host_path(self, host_path: Path) -> str:
        resolved_host_path = host_path.resolve()
        relative_path = resolved_host_path.relative_to(self.mount_host_root)
        return str(Path(self.mount_container_path) / relative_path)

    def execute_notebook(self, notebook_host_path: Path, timeout_seconds: int) -> ExecutionAttemptResult:
        if self.container is None:
            raise RecoverableNotebookExecutionError("Container is not running")

        container_notebook_path = self.container_path_for_host_path(notebook_host_path)
        command = [
            "docker",
            "exec",
            self.container_name,
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            container_notebook_path,
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RecoverableNotebookExecutionError(
                f"Notebook execution exceeded timeout of {timeout_seconds} seconds"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RecoverableNotebookExecutionError(f"Failed to execute notebook: {exc}") from exc

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            self._reload_container_state()
            if not self._container_is_running():
                raise RecoverableNotebookExecutionError(
                    "Container stopped while executing the notebook"
                )

            if self._looks_like_notebook_failure(stdout, stderr):
                raise NonRecoverableNotebookExecutionError(
                    "Notebook execution failed due to a notebook error"
                )

            raise RecoverableNotebookExecutionError(
                f"Notebook execution returned non-zero exit code {completed.returncode}"
            )

        return ExecutionAttemptResult(
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            timed_out=False,
        )

    def _reload_container_state(self) -> None:
        try:
            if self.container is not None:
                self.container.reload()
        except Exception:
            pass

    def _container_is_running(self) -> bool:
        if self.container is None:
            return False
        try:
            self.container.reload()
            return self.container.status == "running"
        except Exception:
            return False

    @staticmethod
    def _looks_like_notebook_failure(stdout: str, stderr: str) -> bool:
        combined = f"{stdout}\n{stderr}"
        markers = (
            "CellExecutionError",
            "nbconvert failed",
            "Traceback",
            "error while executing",
            "Error executing",
        )
        return any(marker in combined for marker in markers)


def load_runner_config(
    setting: str,
    model: str,
    results_root: Path,
    juno_bench_root: Path,
    config_spec: Path = DEFAULT_CONFIG_PATH,
    defaults_spec: Path = DEFAULT_DEFAULTS_PATH,
    max_retries: int = 3,
    notebook_timeout: int | None = None,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    overwrite: bool = False,
) -> RunnerConfig:
    config = load_config(config_spec, defaults_spec)
    environment = config.get("environment", {})

    resolved_timeout = notebook_timeout if notebook_timeout is not None else int(environment.get("timeout", 1800))

    return RunnerConfig(
        setting=setting,
        model=model,
        results_root=results_root.resolve(),
        juno_bench_root=juno_bench_root.resolve(),
        benchmark_root=(juno_bench_root / "benchmark").resolve(),
        temp_root=DEFAULT_TMP_ROOT.resolve(),
        container_mount_path=DEFAULT_CONTAINER_MOUNT_PATH,
        docker_image_name=str(environment.get("docker_image_name", "yarinamomo/kaggle_python_env")),
        docker_start_command=environment.get("docker_start_command"),
        port=int(environment.get("port", DEFAULT_PORT)),
        notebook_timeout=resolved_timeout,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        overwrite=overwrite,
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


def find_benchmark_test_notebook(benchmark_root: Path, instance_name: str) -> Path:
    """Find the benchmark notebook that contains the validation test cell.

    Canonical source is strictly <instance>_fixed_with_test_executed.ipynb.
    """
    instance_dir = benchmark_root / instance_name
    if not instance_dir.exists():
        raise NotebookAssemblyError(f"Benchmark instance directory not found: {instance_dir}")

    canonical_path = instance_dir / f"{instance_name}_fixed_with_test_executed.ipynb"
    if canonical_path.exists():
        return canonical_path

    raise NotebookAssemblyError(
        "Required benchmark notebook missing: "
        f"{canonical_path}"
    )


def load_last_test_cell_source(instance_name: str, benchmark_root: Path) -> str:
    notebook_path = find_benchmark_test_notebook(benchmark_root, instance_name)

    notebook = nbformat.read(notebook_path, as_version=4)
    if not notebook.cells:
        raise NotebookAssemblyError(f"Validated test notebook has no cells: {notebook_path}")

    last_cell = notebook.cells[-1]
    source = last_cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    if not isinstance(source, str) or not source.strip():
        raise NotebookAssemblyError(f"Last cell source is empty in notebook: {notebook_path}")

    return source


def build_execution_notebook(patched_path: Path, benchmark_root: Path) -> nbformat.NotebookNode:
    patch_source = patched_path.read_text(encoding="utf-8")
    patch_cells = _patched_script_to_cells(patch_source)
    if not patch_cells:
        raise NotebookAssemblyError(f"No code cells parsed from patched script: {patched_path}")

    instance_name = extract_instance_name_from_patched_path(patched_path)
    test_cell_source = load_last_test_cell_source(instance_name, benchmark_root)

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source=source) for source in patch_cells]
    notebook.cells.append(nbformat.v4.new_code_cell(source=test_cell_source))
    return notebook


def write_notebook(notebook: nbformat.NotebookNode, notebook_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, notebook_path)


def prepare_execution_notebook(
    patched_path: Path,
    prepared_path: Path,
    benchmark_root: Path,
) -> None:
    notebook = build_execution_notebook(patched_path, benchmark_root)
    write_notebook(notebook, prepared_path)


def stage_instance_workspace(instance_name: str, benchmark_root: Path, staged_instance_dir: Path) -> None:
    """Copy one benchmark instance folder into a temp execution workspace."""
    source_instance_dir = benchmark_root / instance_name
    if not source_instance_dir.exists():
        raise NotebookAssemblyError(f"Benchmark instance directory not found: {source_instance_dir}")

    if staged_instance_dir.exists():
        shutil.rmtree(staged_instance_dir)

    shutil.copytree(source_instance_dir, staged_instance_dir)


def copy_executed_notebook(prepared_path: Path, output_path: Path) -> None:
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared notebook not found: {prepared_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prepared_path, output_path)


def _code_cell_has_error_output(cell: nbformat.NotebookNode) -> bool:
    outputs = cell.get("outputs", [])
    return any(output.get("output_type") == "error" for output in outputs)


def _analyze_test_cell_failure(notebook_path: Path) -> tuple[bool, bool]:
    """Return (test_cell_error, only_test_cell_error) for an executed notebook."""
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.get("cell_type") == "code"]
    if not code_cells:
        return False, False

    error_indices = [idx for idx, cell in enumerate(code_cells) if _code_cell_has_error_output(cell)]
    if not error_indices:
        return False, False

    test_cell_index = len(code_cells) - 1
    test_cell_error = test_cell_index in error_indices
    only_test_cell_error = test_cell_error and len(error_indices) == 1
    return test_cell_error, only_test_cell_error


def _set_record_test_failure_flags(record: NotebookRunRecord, notebook_path: Path) -> None:
    try:
        test_cell_error, only_test_cell_error = _analyze_test_cell_failure(notebook_path)
        record.test_cell_error = test_cell_error
        record.only_test_cell_error = only_test_cell_error
    except Exception:
        # Keep defaults when notebook cannot be analyzed.
        pass


def _analyze_execution_proof(notebook_path: Path) -> tuple[bool, str]:
    """Return whether the notebook contains evidence of execution."""
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.get("cell_type") == "code"]
    if not code_cells:
        return False, "Notebook has no code cells"

    execution_count_cells = sum(1 for cell in code_cells if cell.get("execution_count") is not None)
    output_cells = sum(1 for cell in code_cells if len(cell.get("outputs", [])) > 0)

    if execution_count_cells > 0:
        return True, f"{execution_count_cells} code cells have execution_count"

    if output_cells > 0:
        return True, f"{output_cells} code cells contain outputs"

    return False, "No execution_count and no outputs found"


def _set_record_execution_proof(record: NotebookRunRecord, notebook_path: Path) -> None:
    try:
        has_proof, detail = _analyze_execution_proof(notebook_path)
        record.execution_proof = has_proof
        record.execution_proof_detail = detail
    except Exception as exc:  # noqa: BLE001
        record.execution_proof = False
        record.execution_proof_detail = f"Execution proof check failed: {exc}"


def build_output_path(patched_path: Path) -> Path:
    return patched_path.with_name(f"{patched_path.stem}_executed.ipynb")


def build_prepared_path(
    patched_path: Path,
    config: RunnerConfig,
    attempt: int,
) -> Path:
    run_name = patched_path.parent.name
    instance_name = extract_instance_name_from_patched_path(patched_path)
    unique_token = uuid.uuid4().hex[:8]
    return (
        config.temp_root
        / config.setting
        / config.model
        / run_name
        / instance_name
        / f"attempt_{attempt}_{unique_token}"
        / instance_name
        / f"{instance_name}_patched_with_test.ipynb"
    )


def process_single_patched_notebook(patched_path: Path, config: RunnerConfig) -> NotebookRunRecord:
    run_name = patched_path.parent.name
    instance_name = extract_instance_name_from_patched_path(patched_path)
    output_path = build_output_path(patched_path)
    last_record = NotebookRunRecord(
        setting=config.setting,
        model=config.model,
        run_name=run_name,
        instance_name=instance_name,
        patched_path=str(patched_path),
        prepared_notebook_path="",
        executed_notebook_path="",
        output_path=str(output_path),
        status="not_started",
        attempts=0,
        message="",
    )

    for attempt in range(1, config.max_retries + 1):
        prepared_path = build_prepared_path(patched_path, config, attempt)
        last_record.prepared_notebook_path = str(prepared_path)
        last_record.attempts = attempt

        try:
            staged_instance_dir = prepared_path.parent
            stage_instance_workspace(instance_name, config.benchmark_root, staged_instance_dir)
            prepare_execution_notebook(patched_path, prepared_path, config.benchmark_root)
        except Exception as exc:  # noqa: BLE001
            last_record.status = "assembly_failed"
            last_record.message = str(exc)
            return last_record

        executor: DockerNotebookExecutor | None = None

        try:
            executor = DockerNotebookExecutor(
                image_name=config.docker_image_name,
                start_command=config.docker_start_command,
                mount_host_root=config.temp_root,
                mount_container_path=config.container_mount_path,
                port=config.port,
            )
            with executor:
                execution_result = executor.execute_notebook(prepared_path, config.notebook_timeout)
                last_record.stdout = execution_result.stdout
                last_record.stderr = execution_result.stderr
                last_record.executed_notebook_path = str(prepared_path)
                _set_record_execution_proof(last_record, prepared_path)
                _set_record_test_failure_flags(last_record, prepared_path)
                if not last_record.execution_proof:
                    raise RecoverableNotebookExecutionError(
                        f"Notebook lacks execution proof after run: {last_record.execution_proof_detail}"
                    )
                copy_executed_notebook(prepared_path, output_path)
                last_record.status = "completed"
                last_record.message = "Notebook executed successfully"
                return last_record
        except NonRecoverableNotebookExecutionError as exc:
            last_record.executed_notebook_path = str(prepared_path)
            if prepared_path.exists():
                _set_record_execution_proof(last_record, prepared_path)
                _set_record_test_failure_flags(last_record, prepared_path)
                if last_record.execution_proof:
                    copy_executed_notebook(prepared_path, output_path)
            last_record.status = "notebook_failed"
            last_record.message = str(exc)
            return last_record
        except RecoverableNotebookExecutionError as exc:
            last_record.executed_notebook_path = str(prepared_path)
            last_record.status = "retrying"
            last_record.message = str(exc)
            if prepared_path.exists():
                try:
                    _set_record_execution_proof(last_record, prepared_path)
                    _set_record_test_failure_flags(last_record, prepared_path)
                    if last_record.execution_proof:
                        copy_executed_notebook(prepared_path, output_path)
                except Exception:
                    pass
            if attempt < config.max_retries:
                time.sleep(config.retry_delay_seconds)
                continue
            last_record.status = "container_or_timeout_failed"
            return last_record
        finally:
            if executor is not None:
                executor.stop()

    return last_record


def write_report(report_path: Path, config: RunnerConfig, records: list[NotebookRunRecord]) -> None:
    status_counts = Counter(record.status for record in records)
    report_summary = {
        "total": len(records),
        "status_counts": dict(status_counts),
        "test_cell_error_count": sum(1 for record in records if record.test_cell_error),
        "only_test_cell_error_count": sum(1 for record in records if record.only_test_cell_error),
        "execution_proof_count": sum(1 for record in records if record.execution_proof),
        "missing_execution_proof_count": sum(1 for record in records if not record.execution_proof),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "setting": config.setting,
        "model": config.model,
        "results_root": str(config.results_root),
        "juno_bench_root": str(config.juno_bench_root),
        "benchmark_root": str(config.benchmark_root),
        "temp_root": str(config.temp_root),
        "summary": report_summary,
        "records": [asdict(record) for record in records],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_all_patched_notebooks(config: RunnerConfig) -> dict[str, Any]:
    patched_paths = discover_patched_notebooks(config.results_root, config.setting, config.model)
    records: list[NotebookRunRecord] = []

    progress_desc = f"Executing patched notebooks [{config.setting}/{config.model}]"
    for patched_path in tqdm(patched_paths, desc=progress_desc, unit="notebook"):
        output_path = build_output_path(patched_path)
        if output_path.exists() and not config.overwrite:
            skipped_record = NotebookRunRecord(
                setting=config.setting,
                model=config.model,
                run_name=patched_path.parent.name,
                instance_name=extract_instance_name_from_patched_path(patched_path),
                patched_path=str(patched_path),
                prepared_notebook_path="",
                executed_notebook_path=str(output_path),
                output_path=str(output_path),
                status="skipped_existing",
                attempts=0,
                message="Executed notebook already exists",
            )
            _set_record_execution_proof(skipped_record, output_path)
            _set_record_test_failure_flags(skipped_record, output_path)
            records.append(skipped_record)
            continue
        
        print(f"Processing {patched_path}...")
        records.append(process_single_patched_notebook(patched_path, config))
        print(f"Finished {patched_path} with status {records[-1].status} after {records[-1].attempts} attempts")

    report_path = config.results_root / config.setting / config.model / "analysis" / "patched_notebook_execution_report.json"
    write_report(report_path, config, records)

    return {
        "report_path": str(report_path),
        "records": [asdict(record) for record in records],
    }


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


def run_all_settings_with_defaults() -> dict[str, Any]:
    results_root = DEFAULT_RESULTS_ROOT
    juno_bench_root = DEFAULT_JUNO_BENCH_ROOT
    setting_model_pairs = discover_setting_model_pairs(results_root)
    if not setting_model_pairs:
        raise FileNotFoundError(f"No patched notebooks found under: {results_root}")

    all_runs: list[dict[str, Any]] = []
    for setting, model in setting_model_pairs:
        print(f"Starting setting/model: {setting}/{model}")
        config = load_runner_config(
            setting=setting,
            model=model,
            results_root=results_root,
            juno_bench_root=juno_bench_root,
            config_spec=DEFAULT_CONFIG_PATH,
            defaults_spec=DEFAULT_DEFAULTS_PATH,
            max_retries=DEFAULT_MAX_RETRIES,
            notebook_timeout=None,
            retry_delay_seconds=DEFAULT_RETRY_DELAY_SECONDS,
            overwrite=False,
        )
        run_result = run_all_patched_notebooks(config)
        all_runs.append(
            {
                "setting": setting,
                "model": model,
                "report_path": run_result["report_path"],
                "record_count": len(run_result["records"]),
            }
        )
        print(f"Finished setting/model: {setting}/{model}")

    summary_path = results_root / "analysis" / "all_settings_patched_notebook_execution_report.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "results_root": str(results_root.resolve()),
        "juno_bench_root": str(juno_bench_root.resolve()),
        "total_setting_model_runs": len(all_runs),
        "runs": all_runs,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "summary_report_path": str(summary_path),
        "runs": all_runs,
    }


def main() -> dict[str, Any]:
    return run_all_settings_with_defaults()


if __name__ == "__main__":
    main()
