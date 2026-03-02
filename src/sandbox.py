from typing import Optional, cast, Dict, Any
import queue
import time
import requests
import docker
import websocket
import json
import uuid
import threading
from src.utils.log import logger
from src.utils.nb_types import CellExecutionResult
import os
import datetime

class DockerSandbox:
    def __init__(self, image_name, base_url="http://127.0.0.1", port=8888, token="Super_Duper_Secret_Token", mount_volume=None, start_command=None):
        self.image_name = image_name
        self.base_url = base_url
        self.port = port
        self.token = token
        self.mount_volume = mount_volume
        self.docker_client = docker.from_env()
        self.container = None
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.kernel_id: Optional[str] = None
        self.session_id = str(uuid.uuid4())
        self.execution_results: Dict[str, Dict[str, Any]] = {}
        self.ws_lock = threading.Lock()
        self.start_command = start_command

    def start(self):
        # Remove old container if exists
        try:
            old = self.docker_client.containers.get(f"sandbox_{self.port}")
            old.remove(force=True)
            time.sleep(0.5)  # Wait for Docker to fully release the container name
        except Exception:
            pass

        docker_kwargs: Dict[str, Any] = dict(
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
        self._start_kernel_websocket()

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

    def _start_kernel_websocket(self):
        """Create a new kernel and establish WebSocket connection."""
        try:
            # Create new kernel via REST API
            resp = requests.post(
                f"{self.base_url}:{self.port}/api/kernels",
                headers={"Authorization": f"Token {self.token}"},
                json={},
                timeout=10
            )
            resp.raise_for_status()
            kernel_info = resp.json()
            self.kernel_id = kernel_info['id']
            logger.info(f"Created kernel: {self.kernel_id}")
            
            # Connect WebSocket
            ws_url = f"ws://127.0.0.1:{self.port}/api/kernels/{self.kernel_id}/channels?token={self.token}"
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # Run WebSocket in background thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()
            
            # Wait for WebSocket to actually connect (not just started)
            self._wait_for_websocket_connection(timeout=10)
            logger.info("WebSocket connected for kernel %s", self.kernel_id)
            
        except Exception as exc:
            logger.exception("Failed to start kernel WebSocket")
            raise RuntimeError("Failed to start kernel") from exc
    
    def _wait_for_websocket_connection(self, timeout=10):
        """Wait for WebSocket to be properly connected."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ws and self.ws.sock and self.ws.sock.connected:
                logger.debug("WebSocket connection verified")
                return
            time.sleep(0.1)
        raise RuntimeError("WebSocket failed to establish connection within timeout")

    def _on_ws_message(self, ws, message):
        """Handle incoming WebSocket messages from kernel."""
        try:
            msg = json.loads(message)
            msg_type = msg.get('msg_type')
            content = msg.get('content', {})
            parent_msg_id = msg.get('parent_header', {}).get('msg_id')
            
            if parent_msg_id not in self.execution_results:
                return
            
            result = self.execution_results[parent_msg_id]
            
            # Accumulate outputs from different message types
            if msg_type == 'stream':
                result['outputs'].append({
                    'output_type': 'stream',
                    'name': content.get('name', 'stdout'),
                    'text': content.get('text', '')
                })
            elif msg_type == 'execute_result':
                result['outputs'].append({
                    'output_type': 'execute_result',
                    'data': content.get('data', {}),
                    'execution_count': content.get('execution_count')
                })
            elif msg_type == 'display_data':
                result['outputs'].append({
                    'output_type': 'display_data',
                    'data': content.get('data', {}),
                    'metadata': content.get('metadata', {})
                })
            elif msg_type == 'error':
                result['outputs'].append({
                    'output_type': 'error',
                    'ename': content.get('ename', ''),
                    'evalue': content.get('evalue', ''),
                    'traceback': content.get('traceback', [])
                })
            elif msg_type == 'execute_reply':
                result['status'] = content.get('status', 'unknown')
                result['execution_count'] = content.get('execution_count')
                result['done'] = True
                logger.debug("Execution %s completed with status: %s", parent_msg_id, result['status'])
                
        except Exception as e:
            logger.exception("Error handling WebSocket message")
            logger.exception(e) 

    def _on_ws_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        self.ws = None

    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure."""
        logger.info("WebSocket closed (code: %s, msg: %s)", close_status_code, close_msg)
        self.ws = None

    def run(self, code: str, timeout=30, max_retries=2) -> CellExecutionResult:
        """
        Execute code in the container kernel and return outputs.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds before interrupting (default: 30)
            max_retries: Maximum reconnection attempts (default: 2)
            
        Returns:
            CellExecutionResult with execution output
            
        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If kernel is unresponsive or execution fails
        """
        # Ensure WebSocket is connected with retries
        for attempt in range(max_retries + 1):
            if not self._is_websocket_connected():
                if attempt == 0:
                    logger.warning("WebSocket disconnected, attempting to reconnect...")
                else:
                    logger.info("Reconnection attempt %d/%d", attempt, max_retries)
                
                try:
                    self._start_kernel_websocket()
                except Exception as e:
                    if attempt == max_retries:
                        raise RuntimeError(f"Failed to reconnect to kernel after {max_retries} attempts") from e
                    time.sleep(1)
                    continue
            
            # WebSocket is connected, attempt execution
            try:
                return self._execute_code(code, timeout)
            except RuntimeError as e:
                # If WebSocket disconnected during execution, try reconnecting
                if "WebSocket disconnected" in str(e) and attempt < max_retries:
                    logger.warning("WebSocket disconnected during execution, retrying...")
                    time.sleep(1)
                    continue
                # Otherwise, re-raise the error
                raise
        
        raise RuntimeError("Failed to execute code after all retry attempts")
    
    def _is_websocket_connected(self) -> bool:
        """Check if WebSocket is properly connected."""
        try:
            return (
                self.ws is not None 
                and self.ws.sock is not None 
                and self.ws.sock.connected
            )
        except Exception:
            return False
    
    def _execute_code(self, code: str, timeout=30) -> CellExecutionResult:
        """Internal method to execute code without retry logic."""
        if not self._is_websocket_connected():
            raise RuntimeError("WebSocket disconnected before execution")
        
        msg_id = str(uuid.uuid4())
        self.execution_results[msg_id] = {
            'status': 'pending',
            'outputs': [],
            'done': False,
            'execution_count': None
        }
        
        try:
            # Send execute request
            execute_msg = {
                'header': {
                    'msg_id': msg_id,
                    'msg_type': 'execute_request',
                    'session': self.session_id,
                    'username': 'user',
                    'date': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                },
                'metadata': {},
                'content': {
                    'code': code,
                    'silent': False,
                    'store_history': True,
                    'user_expressions': {},
                    'allow_stdin': False,
                    'stop_on_error': True
                },
                'buffers': [],
                'parent_header': {}
            }
            
            with self.ws_lock:
                if not self._is_websocket_connected():
                    raise RuntimeError("WebSocket disconnected before sending execution request")
                self.ws.send(json.dumps(execute_msg)) # type: ignore - ws is set due to lock and connection check
            
            logger.debug(f"Sent execute request: {msg_id}")
            
            # Wait for execution to complete with timeout
            start_time = time.time()
            while not self.execution_results[msg_id]['done']:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.warning("Execution timeout after %.1f seconds, interrupting kernel...", elapsed)
                    
                    try:
                        self._interrupt_kernel()
                    except Exception as e:
                        logger.exception("Failed to interrupt kernel")
                    
                    # Wait a bit for interrupt to take effect
                    time.sleep(1)
                    
                    # Mark as done to stop waiting
                    self.execution_results[msg_id]['done'] = True
                    self.execution_results[msg_id]['status'] = 'timeout'
                    
                    # Clean up and restart kernel
                    try:
                        self._restart_kernel_websocket()
                    except Exception as e:
                        logger.exception("Failed to restart kernel after timeout")
                    
                    raise TimeoutError(f"Code execution exceeded {timeout} seconds and was interrupted")
                
                time.sleep(0.05)
            
            result = self.execution_results.pop(msg_id)
            
            return cast(CellExecutionResult, {
                'status': result['status'],
                'outputs': result['outputs'],
                'execution_count': result['execution_count'],
            })
            
        except TimeoutError:
            self.execution_results.pop(msg_id, None)
            raise
        except Exception as exc:
            self.execution_results.pop(msg_id, None)
            logger.exception("Execution failed: %s", exc)
            raise RuntimeError("Kernel execution failed") from exc

    def _interrupt_kernel(self):
        """Send interrupt signal to the kernel via REST API."""
        try:
            if not self.kernel_id:
                raise RuntimeError("No kernel ID available")
            
            resp = requests.post(
                f"{self.base_url}:{self.port}/api/kernels/{self.kernel_id}/interrupt",
                headers={"Authorization": f"Token {self.token}"},
                timeout=5
            )
            resp.raise_for_status()
            logger.info("Interrupt signal sent to kernel %s", self.kernel_id)
        except Exception as e:
            logger.exception("Failed to interrupt kernel: %s", e)
            raise

    def _restart_kernel_websocket(self):
        """Cleanly shutdown kernel and restart it with fresh WebSocket connection."""
        try:
            # Close WebSocket
            if self.ws:
                with self.ws_lock:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
            
            # Wait for thread to finish
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2)
            
            # Delete old kernel via REST API
            if self.kernel_id:
                try:
                    requests.delete(
                        f"{self.base_url}:{self.port}/api/kernels/{self.kernel_id}",
                        headers={"Authorization": f"Token {self.token}"},
                        timeout=5
                    )
                    logger.info("Deleted old kernel: %s", self.kernel_id)
                except Exception as e:
                    logger.warning("Failed to delete old kernel: %s", e)
                
                self.kernel_id = None
            
            # Clear pending execution results
            self.execution_results.clear()
            
            # Start fresh kernel
            self._start_kernel_websocket()
            logger.info("Kernel restarted successfully")
            
        except Exception as e:
            logger.exception("Failed to restart kernel")
            raise RuntimeError("Failed to restart kernel") from e
    
    def restart_kernel(self):
        """Public method to restart the kernel without stopping the container."""
        self._restart_kernel_websocket()

    def stop(self):
        """Stop kernel and remove container."""
        try:
            if self.ws:
                with self.ws_lock:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
            
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2)
            
            if self.kernel_id:
                try:
                    requests.delete(
                        f"{self.base_url}:{self.port}/api/kernels/{self.kernel_id}",
                        headers={"Authorization": f"Token {self.token}"},
                        timeout=5
                    )
                except Exception as e:
                    logger.warning("Error deleting kernel: %s", e)
                self.kernel_id = None
        except Exception as e:
            logger.exception("Error stopping kernel: %s", e)
        
        try:
            if self.container:
                self.container.kill()
                self.container.remove()
                self.container = None
        except Exception as e:
            logger.exception("Error stopping container: %s", e)