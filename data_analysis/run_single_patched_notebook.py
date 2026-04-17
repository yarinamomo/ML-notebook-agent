from __future__ import annotations

import docker
from docker.models.containers import Container
import os
import shlex
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.util_test_cells_evaluation import (
    VALIDATED_TEST_NOTEBOOK_EXCLUSIONS,
    _patched_script_to_cells,
)
from src.utils.yaml_parser import load_config


# Hardcoded run inputs: edit patched script and instance folder for your run.
PATCHED_SCRIPT_PATH = PROJECT_ROOT / "results" / "agent" / "glm-4.7-355b" / "run_1" / "numpy_1_patched.py"
INSTANCE_FOLDER_PATH = PROJECT_ROOT / "JunoBench" / "benchmark" / "numpy_1"


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

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

def _load_runner_environment() -> dict[str, Any]:
    _load_dotenv(PROJECT_ROOT / ".env")

    config_path = Path(os.getenv("NOTEBOOK_AGENT_CONFIG_PATH", "../config/agent.yaml"))
    defaults_path = Path(os.getenv("NOTEBOOK_AGENT_DEFAULTS_PATH", "../config/defaults.yaml"))
    return load_config(config_path, defaults_path).get("environment", {})



def _decode_output(value: bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace")


class DockerNotebookExecutor:
    """Minimal single-notebook executor.

    - Init with one host mount directory.
    - Call execute(mounted_folder, notebook_name).
    """

    def __init__(self, mount_host_dir: Path) -> None:
        environment = _load_runner_environment()
        self.mount_host_dir = mount_host_dir.resolve()
        self.mount_tmp_dir = (self.mount_host_dir / "_runtime_tmp").resolve()
        self.mount_container_dir = "/app/container"
        self.image_name = str(environment.get("docker_image_name", "yarinamomo/kaggle_python_env"))
        self.timeout_seconds = int(environment.get("timeout", 1800))
        self.container_name = f"single-notebook-runner-{uuid.uuid4().hex[:8]}"
        self.client = docker.from_env()
        self.container: Container | None = None

    def _container_environment(self) -> dict[str, str]:
        return {
            "HOME": self.mount_container_dir,
            "TMPDIR": "/tmp",
            "JUPYTER_CONFIG_DIR": "/tmp/jupyter-config",
            "JUPYTER_DATA_DIR": "/tmp/jupyter-data",
            "JUPYTER_RUNTIME_DIR": "/tmp/jupyter-runtime",
            "MPLCONFIGDIR": "/tmp/mplconfig",
            "IPYTHONDIR": "/tmp/ipython",
            "XDG_CONFIG_HOME": "/tmp/xdg-config",
            "XDG_DATA_HOME": "/tmp/xdg-data",
            "XDG_CACHE_HOME": "/tmp/xdg-cache",
        }

    def __enter__(self) -> "DockerNotebookExecutor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self.mount_host_dir.mkdir(parents=True, exist_ok=True)
        self.mount_tmp_dir.mkdir(parents=True, exist_ok=True)

        docker_kwargs = {
            "name": self.container_name,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "command": ["bash", "-lc", "sleep infinity"],
            "volumes": {
                str(self.mount_host_dir): {
                    "bind": self.mount_container_dir,
                    "mode": "rw",
                },
                str(self.mount_tmp_dir): {
                    "bind": "/tmp",
                    "mode": "rw",
                }
            },
            "environment": self._container_environment(),
            "user": "0",
        }

        self.container = self.client.containers.run(self.image_name, **docker_kwargs)

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

    def execute(self, mounted_folder: Path, notebook_name: str) -> tuple[int, str, str]:
        if self.container is None:
            raise RuntimeError("Container is not running")

        mounted_folder = mounted_folder.resolve()
        relative_dir = mounted_folder.relative_to(self.mount_host_dir)
        container_workdir = str(Path(self.mount_container_dir) / relative_dir)

        command = (
            "set -eu; "
            "mkdir -p /tmp/jupyter-config /tmp/jupyter-data /tmp/jupyter-runtime /tmp/mplconfig /tmp/ipython /tmp/xdg-config /tmp/xdg-data /tmp/xdg-cache; "
            "touch /tmp/kaggle.log; "
            "jupyter nbconvert --Application.ignore_config=True --to notebook --execute --inplace "
            "--ExecutePreprocessor.allow_errors=True "
            f"--ExecutePreprocessor.timeout={self.timeout_seconds} {shlex.quote(notebook_name)}"
        )

        result = self.container.exec_run(
            cmd=["bash", "-lc", command],
            workdir=container_workdir,
            environment=self._container_environment(),
            demux=True,
        )

        stdout = ""
        stderr = ""
        if isinstance(result.output, tuple):
            stdout = _decode_output(result.output[0])
            stderr = _decode_output(result.output[1])
        else:
            stdout = _decode_output(result.output)

        return int(result.exit_code), stdout, stderr


def _validate_inputs(patched_script: Path, instance_folder: Path) -> None:
    if not patched_script.exists():
        raise FileNotFoundError(f"Patched script not found: {patched_script}")
    if not instance_folder.exists():
        raise FileNotFoundError(f"Instance folder not found: {instance_folder}")
    if not patched_script.is_file():
        raise FileNotFoundError(f"Patched script is not a file: {patched_script}")
    if not instance_folder.is_dir():
        raise FileNotFoundError(f"Instance folder is not a directory: {instance_folder}")


def _reset_tmp_workdir(tmp_root: Path) -> None:
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)


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


def run_single_patched_notebook(patched_script: Path, instance_folder: Path) -> None:
    environment = _load_runner_environment()
    configured_mount = Path(str(environment.get("docker_mount_path", "tmp/single_notebook_run")))
    tmp_root = configured_mount if configured_mount.is_absolute() else PROJECT_ROOT / configured_mount
    tmp_root = tmp_root.resolve()

    _validate_inputs(patched_script, instance_folder)
    _reset_tmp_workdir(tmp_root)

    staged_instance_dir, staged_patched_script = _stage_inputs(patched_script, instance_folder, tmp_root)
    staged_notebook_path = _build_notebook_from_patched_script(staged_patched_script, staged_instance_dir)

    output_notebook_path = patched_script.with_name(f"{patched_script.stem}_executed.ipynb")

    print(f"Staged instance dir: {staged_instance_dir}")
    print(f"Staged notebook: {staged_notebook_path}")
    print(f"Executing notebook in Docker...")
    with DockerNotebookExecutor(mount_host_dir=tmp_root) as executor:
        exit_code, stdout, stderr = executor.execute(
            mounted_folder=staged_instance_dir,
            notebook_name=staged_notebook_path.name,
        )
    
    output_notebook_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Executed notebook copied to: {output_notebook_path}")
    shutil.copy2(staged_notebook_path, output_notebook_path)

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"Removed tmp workdir: {tmp_root}")
    
    if exit_code != 0:
        raise RuntimeError(
            "Notebook execution failed\n"
            f"returncode={exit_code}\n\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{stderr}"
        )



if __name__ == "__main__":
    PATCHED_SCRIPT_PATH = PROJECT_ROOT / "results" / "agent" / "glm-4.7-355b" / "run_1" / "numpy_1_patched.py"
    INSTANCE_FOLDER_PATH = PROJECT_ROOT / "JunoBench" / "benchmark" / "numpy_1"
    run_single_patched_notebook(
        patched_script=PATCHED_SCRIPT_PATH,
        instance_folder=INSTANCE_FOLDER_PATH
    )
