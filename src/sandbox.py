from typing import Optional, cast
import queue
import time
import requests
import docker
from jupyter_kernel_client import KernelClient
from src.utils.log import logger
from src.utils.nb_types import CellExecutionResult
import os

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
        # self._closed = False
        self.start_command = start_command

    def start(self):
        # Remove old container if exists
        try:
            old = self.docker_client.containers.get(f"sandbox_{self.port}")
            old.remove(force=True)
        except Exception:
            pass

        docker_kwargs = dict(
            name=f"sandbox_{self.port}",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={self.mount_volume: {'bind': '/app/container', 'mode': 'rw'}} if self.mount_volume else None,
            ports={'8888/tcp': self.port},
            command=self.start_command,
            environment={"HOME": "/app/container"}
        )

        # Only set user on Unix systems to give write permissions to mounted volume without needing to change permissions on host
        if os.name == "posix":
            docker_kwargs["user"] = f"{os.getuid()}:{os.getgid()}"

        self.container = self.docker_client.containers.run(
            self.image_name,
            **docker_kwargs
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
                logger.warning("Jupyter status check failed: %s", exc)
                time.sleep(1)
        raise RuntimeError("Jupyter server not responding")

    def _start_kernel(self):
        try:
            kernel_client = KernelClient(server_url=f"{self.base_url}:{self.port}", token=self.token)
            kernel_client.start()
            self.kernel_client = kernel_client
        except Exception as exc:
            logger.exception("Failed to start kernel")
            raise RuntimeError("Failed to start kernel") from exc

    def run(self, code: str, max_retries=2) -> CellExecutionResult:
        """
        Execute code in the container kernel and return outputs.
        Automatically attempts to reconnect the kernel if disconnected before execution.
        
        Args:
            code: Python code to execute
            max_retries: Maximum number of reconnection attempts if kernel is disconnected (default: 2)
            
        Returns:
            CellExecutionResult with execution output
        """
        # if self._closed:
        #     raise RuntimeError("Sandbox is closed")
        
        # Ensure kernel is connected before execution (with retries)
        for attempt in range(max_retries + 1):
            if not self.kernel_client:
                if attempt == 0:
                    logger.warning("Kernel not connected, attempting to reconnect...")
                else:
                    logger.info("Reconnection attempt %d/%d", attempt, max_retries)
                
                try:
                    self._start_kernel()
                    logger.info("Kernel reconnected successfully")
                    break  # Successfully connected
                except Exception as e:
                    if attempt == max_retries:
                        logger.error("Failed to reconnect kernel after %d attempts", max_retries)
                        raise RuntimeError(f"Failed to reconnect kernel after {max_retries} attempts") from e
                    time.sleep(3)
            else:
                break  # Kernel already connected
        
        # Execute code
        try:
            result = self.kernel_client.execute(code)
            return cast(CellExecutionResult, result)
        except (TimeoutError, queue.Empty) as exc:
            logger.error(f"Kernel connection timed out: {exc}")
            raise TimeoutError(f"Kernel connection timed out") from exc
        except (ConnectionError, OSError) as exc:
            # Kernel died during execution - mark as disconnected for next call
            logger.exception("Kernel connectivity lost during execution: %s", exc)
            self.kernel_client = None
            raise RuntimeError("Kernel connection lost during execution") from exc
        except Exception as exc:
            logger.exception("Kernel execution failed")
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
        # self._closed = True

    def restart_kernel(self):
        """Restart the kernel without stopping the container."""
        old_client = self.kernel_client
        self.kernel_client = None  # Clear before stopping
        
        if old_client:
            try:
                old_client.stop()
            except Exception as e:
                logger.exception("Error stopping old kernel: %s", e)
        
        self._start_kernel()