from typing import Optional, cast
import queue
import time
import requests
import docker
from jupyter_kernel_client import KernelClient
from src.utils.log import logging
from src.utils.nb_types import CellExecutionResult

class DockerSandbox:
    def __init__(self, image_name, base_url="http://127.0.0.1", port=8888, token="Super_Duper_Secret_Token", mount_volume=None, start_command=None):
        self.image_name = image_name
        self.base_url = base_url
        self.port = port
        self.token = token
        self.mount_volume = mount_volume
        self.docker_client = docker.from_env()
        self.container = None
        self.kernel_client: Optional[KernelClient] = None
        self._closed = False
        self.start_command = start_command

    def start(self):
        # Remove old container if exists
        try:
            old = self.docker_client.containers.get(f"sandbox_{self.port}")
            old.remove(force=True)
        except Exception:
            pass

        # Start container with Jupyter server
        self.container = self.docker_client.containers.run(
            self.image_name,
            name=f"sandbox_{self.port}",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={self.mount_volume: {'bind': '/app/container', 'mode': 'rw'}} if self.mount_volume else None,
            ports={'8888/tcp': self.port},
            command=self.start_command
        )

        # Wait until server is ready
        self._wait_for_server()

        # Start kernel and connect WebSocket client
        self._start_kernel()

    def _wait_for_server(self, timeout=30):
        url = f"{self.base_url}:{self.port}/api/status"
        for _ in range(timeout):
            try:
                r = requests.get(url, headers={"Authorization": f"Token {self.token}"}, timeout=2)
                if r.status_code == 200:
                    return
            except Exception as exc:
                logging.warning("Jupyter status check failed: %s", exc)
                time.sleep(1)
        raise RuntimeError("Jupyter server not responding")

    def _start_kernel(self):
        try:
            kernel_client = KernelClient(server_url=f"{self.base_url}:{self.port}", token=self.token)
            kernel_client.start()
            self.kernel_client = kernel_client
        except Exception as exc:
            logging.exception("Failed to start kernel")
            raise RuntimeError("Failed to start kernel") from exc

    def run(self, code: str, timeout=30) -> CellExecutionResult:
        """Execute code in the container kernel and return outputs."""
        if not self.kernel_client or self._closed:
            raise RuntimeError("Kernel not connected" if not self.kernel_client else "Sandbox is closed")
        try:
            print("Executing code in sandbox kernel...")
            print(f"Code:\n{code}")
            result = self.kernel_client.execute(code, timeout=timeout) 
            return cast(CellExecutionResult, result)
        except (TimeoutError, queue.Empty) as exc:
            logging.warning("Kernel execution timed out after %s seconds", timeout)
            raise TimeoutError(f"Kernel execution timed out after {timeout} seconds") from exc
        except Exception as exc:
            logging.exception("Kernel execution failed")
            raise RuntimeError("Kernel execution failed") from exc

    def stop(self):
        """Stop kernel and remove container."""
        if self.kernel_client:
            self.kernel_client.stop()
            self.kernel_client = None
        if self.container:
            self.container.kill()
            self.container.remove()
            self.container = None
        self._closed = True