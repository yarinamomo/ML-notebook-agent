from __future__ import annotations

"""Independent notebook runner utilities.

This module is intentionally independent from the agent and NotebookEnvironment flow.
It only provides a minimal Docker-backed executor for running one notebook.
"""

import os
import shlex
import uuid
from pathlib import Path

import docker
from docker.models.containers import Container
from pathlib import PurePosixPath


def _host_uid_gid() -> tuple[int | None, int | None]:
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return os.getuid(), os.getgid()
    return None, None


def _decode_output(value: bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace")


class DockerNotebookExecutor:
    """Minimal single-notebook Docker executor.

    - Initialize with one host mount directory.
    - Call execute() with a staged folder + notebook filename.
    """

    def __init__(
        self,
        mount_host_dir: Path,
        docker_image_name: str,
        timeout_seconds: int = 1800,
    ) -> None:
        self.mount_host_dir = mount_host_dir.resolve()
        self.mount_container_dir = "/app/container"
        self.image_name = docker_image_name
        self.timeout_seconds = timeout_seconds
        self.host_uid, self.host_gid = _host_uid_gid()
        self.container_name = f"single-notebook-runner-{uuid.uuid4().hex[:8]}"
        self.client = docker.from_env()
        self.container: Container | None = None

    def _container_environment(self) -> dict[str, str]:
        return {
            "HOME": "/tmp/home",
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
        self.container = self.client.containers.run(
            self.image_name,
            name=self.container_name,
            detach=True,
            tty=True,
            stdin_open=True,
            command=["bash", "-lc", "sleep infinity"],
            volumes={
                str(self.mount_host_dir): {
                    "bind": self.mount_container_dir,
                    "mode": "rw",
                }
            },
            environment=self._container_environment(),
            user="0",
        )

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

    def restore_mount_ownership(self) -> None:
        if self.container is None:
            return
        if self.host_uid is None or self.host_gid is None:
            return

        result = self.container.exec_run(
            cmd=[
                "bash",
                "-lc",
                f"chown -R {self.host_uid}:{self.host_gid} {shlex.quote(self.mount_container_dir)}",
            ],
            workdir="/",
            demux=True,
        )

        if int(result.exit_code) != 0:
            stdout = ""
            stderr = ""
            if isinstance(result.output, tuple):
                stdout = _decode_output(result.output[0])
                stderr = _decode_output(result.output[1])
            else:
                stdout = _decode_output(result.output)
            raise RuntimeError(
                "Failed to restore mount ownership inside container\n"
                f"returncode={int(result.exit_code)}\n\n"
                f"stdout:\n{stdout}\n\n"
                f"stderr:\n{stderr}"
            )

    def execute(self, mounted_folder: Path, notebook_name: str) -> tuple[int, str, str]:
        if self.container is None:
            raise RuntimeError("Container is not running")

        mounted_folder = mounted_folder.resolve()
        try:
            relative_dir = mounted_folder.relative_to(self.mount_host_dir)
            # Build a POSIX-style container workdir to avoid Windows backslashes
            container_workdir = str(PurePosixPath(self.mount_container_dir) / PurePosixPath(*relative_dir.parts))
        except Exception:
            # If relative path can't be computed, fall back to the mount root in container
            container_workdir = str(PurePosixPath(self.mount_container_dir))

        command = (
            "set -eu; "
            "mkdir -p /tmp/home /tmp/jupyter-config /tmp/jupyter-data /tmp/jupyter-runtime /tmp/mplconfig /tmp/ipython /tmp/xdg-config /tmp/xdg-data /tmp/xdg-cache; "
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
