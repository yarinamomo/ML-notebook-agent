from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from src.run_nb import docker_executor as de
from src.run_nb import run_single_patched as rsp
from src.run_nb import setup_tmp_folder as stf


def test_validate_inputs_requires_existing_paths(tmp_path: Path):
    patched_script = tmp_path / "missing_patched.py"
    instance_folder = tmp_path / "instance"

    with pytest.raises(FileNotFoundError, match="Patched script not found"):
        rsp._validate_inputs(patched_script, instance_folder)


def test_build_notebook_from_patched_script_appends_test_cell(monkeypatch, tmp_path: Path):
    staged_script = tmp_path / "sample_patched.py"
    staged_script.write_text("irrelevant", encoding="utf-8")

    monkeypatch.setattr(rsp, "_patched_script_to_cells", lambda _source: ["a = 1", "print(a)"])
    monkeypatch.setattr(rsp, "_load_last_benchmark_test_cell_source", lambda _folder: "assert a == 1")

    notebook_path = rsp._build_notebook_from_patched_script(staged_script, tmp_path)
    notebook = nbformat.read(notebook_path, as_version=4)

    assert notebook_path.exists()
    assert len(notebook.cells) == 3
    assert notebook.cells[-1].source == "assert a == 1"


def test_run_single_patched_notebook_success(monkeypatch, tmp_path: Path):
    instance_folder = tmp_path / "demo_instance"
    instance_folder.mkdir()
    patched_script = tmp_path / "demo_patched.py"
    patched_script.write_text("print('patched')", encoding="utf-8")

    monkeypatch.setattr(rsp, "_patched_script_to_cells", lambda _source: ["x = 1"])
    monkeypatch.setattr(rsp, "_load_last_benchmark_test_cell_source", lambda _folder: None)

    calls = {"cleanup": 0}

    def fake_setup(tmp_root: Path, docker_image_name: str):
        tmp_root.mkdir(parents=True, exist_ok=True)

    def fake_cleanup(_tmp_root: Path):
        calls["cleanup"] += 1

    class FakeExecutor:
        def __init__(self, mount_host_dir: Path, docker_image_name: str, timeout_seconds: int):
            self.mount_host_dir = mount_host_dir

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, mounted_folder: Path, notebook_name: str):
            assert mounted_folder.name == "demo_instance"
            assert notebook_name.endswith(".ipynb")
            return 0, "ok", ""

        def restore_mount_ownership(self):
            return None

    monkeypatch.setattr(rsp, "setup_tmp_workdir", fake_setup)
    monkeypatch.setattr(rsp, "cleanup_tmp_workdir", fake_cleanup)
    monkeypatch.setattr(rsp, "DockerNotebookExecutor", FakeExecutor)

    env = {
        "docker_mount_path": str(tmp_path / "mount"),
        "docker_image_name": "test-image",
        "timeout": 10,
        "source_path_parent": str(tmp_path),
    }

    output_path = rsp.run_single_patched_notebook(patched_script, instance_folder.name, env)
    assert output_path is not None
    executed_nb = nbformat.read(output_path, as_version=4)

    assert output_path.exists()
    assert output_path.name == "demo_patched_executed.ipynb"
    assert len(executed_nb.cells) == 1
    assert executed_nb.cells[0].source == "x = 1"
    assert calls["cleanup"] == 1


def test_run_single_patched_notebook_writes_error_file_on_failed_execution(monkeypatch, tmp_path: Path):
    instance_folder = tmp_path / "demo_instance"
    instance_folder.mkdir()
    patched_script = tmp_path / "demo_patched.py"
    patched_script.write_text("print('patched')", encoding="utf-8")

    monkeypatch.setattr(rsp, "_patched_script_to_cells", lambda _source: ["x = 1"])
    monkeypatch.setattr(rsp, "_load_last_benchmark_test_cell_source", lambda _folder: None)

    calls = {"cleanup": 0}

    def fake_setup(tmp_root: Path, docker_image_name: str):
        tmp_root.mkdir(parents=True, exist_ok=True)

    def fake_cleanup(_tmp_root: Path):
        calls["cleanup"] += 1

    class FakeExecutor:
        def __init__(self, mount_host_dir: Path, docker_image_name: str, timeout_seconds: int):
            self.mount_host_dir = mount_host_dir

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, mounted_folder: Path, notebook_name: str):
            return 1, "stdout failure", "stderr failure"

        def restore_mount_ownership(self):
            return None

    monkeypatch.setattr(rsp, "setup_tmp_workdir", fake_setup)
    monkeypatch.setattr(rsp, "cleanup_tmp_workdir", fake_cleanup)
    monkeypatch.setattr(rsp, "DockerNotebookExecutor", FakeExecutor)

    env = {
        "docker_mount_path": str(tmp_path / "mount"),
        "docker_image_name": "test-image",
        "timeout": 10,
        "source_path_parent": str(tmp_path),
    }

    result_path = rsp.run_single_patched_notebook(patched_script, instance_folder.name, env)

    assert result_path is None
    assert calls["cleanup"] == 1
    assert (patched_script.with_name("demo_patched_executed.ipynb")).exists()
    error_path = patched_script.with_name("demo_patched_error.txt")
    assert error_path.exists()
    assert "Notebook execution failed" in error_path.read_text(encoding="utf-8")


def test_run_single_patched_notebook_ignores_cleanup_failure(monkeypatch, tmp_path: Path):
    instance_folder = tmp_path / "demo_instance"
    instance_folder.mkdir()
    patched_script = tmp_path / "demo_patched.py"
    patched_script.write_text("print('patched')", encoding="utf-8")

    monkeypatch.setattr(rsp, "_patched_script_to_cells", lambda _source: ["x = 1"])
    monkeypatch.setattr(rsp, "_load_last_benchmark_test_cell_source", lambda _folder: None)

    def fake_setup(tmp_root: Path, docker_image_name: str):
        tmp_root.mkdir(parents=True, exist_ok=True)

    def fake_cleanup(_tmp_root: Path):
        raise RuntimeError("cleanup failed")

    class FakeExecutor:
        def __init__(self, mount_host_dir: Path, docker_image_name: str, timeout_seconds: int):
            self.mount_host_dir = mount_host_dir

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, mounted_folder: Path, notebook_name: str):
            return 0, "ok", ""

        def restore_mount_ownership(self):
            return None

    monkeypatch.setattr(rsp, "setup_tmp_workdir", fake_setup)
    monkeypatch.setattr(rsp, "cleanup_tmp_workdir", fake_cleanup)
    monkeypatch.setattr(rsp, "DockerNotebookExecutor", FakeExecutor)

    env = {
        "docker_mount_path": str(tmp_path / "mount"),
        "docker_image_name": "test-image",
        "timeout": 10,
        "source_path_parent": str(tmp_path),
    }

    output_path = rsp.run_single_patched_notebook(patched_script, instance_folder.name, env)

    assert output_path is not None
    assert output_path.exists()


def test_execute_requires_started_container(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(de.docker, "from_env", lambda: object())
    executor = de.DockerNotebookExecutor(tmp_path, "image", timeout_seconds=12)

    with pytest.raises(RuntimeError, match="Container is not running"):
        executor.execute(tmp_path, "nb.ipynb")


def test_execute_returns_decoded_streams(monkeypatch, tmp_path: Path):
    class FakeContainer:
        def __init__(self):
            self.last_exec_kwargs = None

        def exec_run(self, **kwargs):
            self.last_exec_kwargs = kwargs
            return SimpleNamespace(exit_code=0, output=(b"stdout text", b"stderr text"))

    monkeypatch.setattr(de.docker, "from_env", lambda: object())
    executor = de.DockerNotebookExecutor(tmp_path, "image", timeout_seconds=99)
    executor.container = FakeContainer() # type: ignore

    mounted_folder = tmp_path / "instance"
    mounted_folder.mkdir(parents=True, exist_ok=True)
    code, stdout, stderr = executor.execute(mounted_folder, "n.ipynb")

    assert code == 0
    assert stdout == "stdout text"
    assert stderr == "stderr text"
    assert "--ExecutePreprocessor.timeout=99" in executor.container.last_exec_kwargs["cmd"][2] # type: ignore


def test_setup_tmp_workdir_handles_permission_error(monkeypatch, tmp_path: Path):
    target = tmp_path / "mount"
    target.mkdir()

    calls = {"chown": 0, "rmtree": 0}
    real_rmtree = stf.shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        calls["rmtree"] += 1
        if calls["rmtree"] == 1:
            raise PermissionError("denied")
        return real_rmtree(path, *args, **kwargs)

    def fake_chown(path: Path, image_name: str, host_uid: int, host_gid: int):
        calls["chown"] += 1

    monkeypatch.setattr(stf.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(stf, "_chown_tree_with_container", fake_chown)
    monkeypatch.setattr(stf, "_host_uid_gid", lambda: (1000, 1000))

    stf.setup_tmp_workdir(target, docker_image_name="image")

    assert target.exists()
    assert calls["chown"] == 1
    assert calls["rmtree"] >= 2


def test_cleanup_tmp_workdir_raises_if_folder_still_exists(monkeypatch, tmp_path: Path):
    target = tmp_path / "mount"
    target.mkdir()

    monkeypatch.setattr(stf.shutil, "rmtree", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Temporary workdir could not be removed"):
        stf.cleanup_tmp_workdir(target)
