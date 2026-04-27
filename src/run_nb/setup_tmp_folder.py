from __future__ import annotations

"""Temporary folder setup helpers for standalone run_nb execution.

This module is intentionally independent from the agent and NotebookEnvironment flow.
"""

import os
import shutil
import uuid
from pathlib import Path

import docker
from docker.models.containers import Container


def _host_uid_gid() -> tuple[int, int]:
    return os.getuid(), os.getgid()


def _decode_output(value: bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace")


def _chown_tree_with_container(path: Path, image_name: str, host_uid: int, host_gid: int) -> None:
    client = docker.from_env()
    container: Container | None = None

    try:
        container = client.containers.run(
            image_name,
            name=f"single-notebook-chown-{uuid.uuid4().hex[:8]}",
            detach=True,
            tty=True,
            stdin_open=True,
            command=["bash", "-lc", "sleep infinity"],
            volumes={
                str(path): {
                    "bind": "/app/container",
                    "mode": "rw",
                }
            },
            user="0",
        )

        result = container.exec_run(
            cmd=["bash", "-lc", f"chown -R {host_uid}:{host_gid} /app/container"],
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
                "Failed to restore mount ownership with root container\n"
                f"returncode={int(result.exit_code)}\n\n"
                f"stdout:\n{stdout}\n\n"
                f"stderr:\n{stderr}"
            )
    finally:
        if container is not None:
            try:
                container.kill()
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass


def setup_tmp_workdir(tmp_root: Path, docker_image_name: str) -> None:
    """Reset temporary mount folder for a run, recovering permissions if needed."""
    if tmp_root.exists():
        try:
            shutil.rmtree(tmp_root)
        except PermissionError:
            host_uid, host_gid = _host_uid_gid()
            _chown_tree_with_container(tmp_root, docker_image_name, host_uid, host_gid)
            shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)


def cleanup_tmp_workdir(tmp_root: Path) -> None:
    """Best-effort cleanup of temporary mount folder with validation."""
    shutil.rmtree(tmp_root, ignore_errors=True)
    if tmp_root.exists():
        raise RuntimeError(f"Temporary workdir could not be removed: {tmp_root}")


__all__ = ["setup_tmp_workdir", "cleanup_tmp_workdir"]
