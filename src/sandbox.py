import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional
import docker
from docker.client import DockerClient
from docker.errors import NotFound as ContainerNotFound
from utils.log import logging
import time
import requests
import json
import uuid
import websocket
import threading
import queue
import atexit
import base64
import re


class SandboxResultType(Enum):
    TEXT = "text"
    IMAGE_PNG = "image/png"
    IMAGE_JPG = "image/jpg"

@dataclass
class ResultPart:
    type: SandboxResultType
    content: Any


class SandboxResult:
    """
    Encapsulates the output of a code execution.
    Behaves like a string for backward compatibility (returning text output),
    but stores rich media (images) in .outputs for multimodal agents.
    """
    def __init__(self, outputs):
        self.outputs = outputs # List of Jupyter protocol output dicts

    @property
    def text(self):
        """Aggregates all text content (stdout, stderr, text/plain)."""
        text_parts = []
        for out in self.outputs:
            msg_type = out['msg_type']
            content = out['content']
            
            if msg_type == 'stream':
                text_parts.append(content['text'])
            elif msg_type in ('execute_result', 'display_data'):
                data = content['data']
                if 'text/plain' in data:
                    text_parts.append(data['text/plain'])
            elif msg_type == 'error':
                text_parts.append(f"\nERROR: {content['evalue']}\n")
        return "".join(text_parts)

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"<SandboxResult: {len(self.outputs)} outputs>"
    

    def llm_compatible(self, if_truncate: bool = False, max_words: int = 500) -> str:
        """Converts outputs into a list of ResultPart for LLM consumption."""
        parts = []
        for out in self.outputs:
            msg_type = out['msg_type']
            content = out['content']
            
            if msg_type == 'stream':
                parts.append(ResultPart(SandboxResultType.TEXT, content['text']))
            elif msg_type in ('execute_result', 'display_data'):
                data = content['data']
                if 'image/png' in data:
                    parts.append(ResultPart(SandboxResultType.IMAGE_PNG, data['image/png']))
                elif 'image/jpeg' in data:
                    parts.append(ResultPart(SandboxResultType.IMAGE_JPG, data['image/jpeg']))

                if 'text/plain' in data:
                    parts.append(ResultPart(SandboxResultType.TEXT, data['text/plain']))
                elif 'text/html' in data:
                    parts.append(ResultPart(SandboxResultType.TEXT, data['text/html']))
            elif msg_type == 'error':
                parts.append(ResultPart(SandboxResultType.TEXT, f"\n---ERROR---: {content['evalue']}\n"))
        outputs = self._result_parts_to_string(parts)
        if if_truncate:
            outputs = self._truncate_cell_output(outputs, max_words=max_words)
        return outputs
    
    def _clean_ansi_codes(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def _result_parts_to_string(self, parts: list) -> str:
        """Convert list of ResultPart to clean string output."""
        text_parts = []
        for part in parts:
            if part.type == SandboxResultType.TEXT:
                text_parts.append(self._clean_ansi_codes(part.content))
        return ''.join(text_parts)
    
    def _truncate_cell_output(self, output: str, max_words: int = 500) -> str:
        """Truncate cell output to a reasonable word limit for LLM processing."""
        words = output.split()
        if len(words) <= max_words:
            return output
        
        truncated = ' '.join(words[:max_words])
        remaining_words = len(words) - max_words
        return f"{truncated}\n\n[OUTPUT TRUNCATED - {remaining_words} more words omitted for brevity]"

    def display(self):
        """Renders outputs in a Jupyter Notebook (for debugging)."""
        try:
            from IPython.display import display, Image
        except ImportError:
            print(self.text)
            return

        for out in self.outputs:
            msg_type = out['msg_type']
            content = out['content']
            
            if msg_type == 'stream':
                print(content['text'], end='')
            elif msg_type in ('execute_result', 'display_data'):
                data = content['data']
                # Prioritize rich media
                if 'image/png' in data:
                    display(Image(data=base64.b64decode(data['image/png']), format='png'))
                elif 'image/jpeg' in data:
                    display(Image(data=base64.b64decode(data['image/jpeg']), format='jpeg'))
                elif 'text/plain' in data:
                    print(data['text/plain'])
            elif msg_type == 'error':
                print(f"---ERROR---: {content['evalue']}")


class ExecutionStatus(Enum):
    """Status of an async execution."""
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result from run_async, includes status and optional cancellation info."""
    status: ExecutionStatus
    result: Optional[SandboxResult] = None
    error: Optional[str] = None


class DockerSandbox:
    def __init__(self, image_name, base_url="http://127.0.0.1", port=8888, token="Super_Duper_Secret_Token", mount_volume=None):
        self.client: DockerClient = docker.from_env()
        self.container = None
        self.port = port
        self.mount_volume = mount_volume
        self.token = token
        self.image_name = image_name
        self.base_url = f"{base_url}:{self.port}"
        self.kernel_id = None

        self.ws_app = None
        self.ws_thread = None
        self.ws_open_event = threading.Event()
        self.execution_queues = {}

        self._queue_lock = threading.Lock()

        atexit.register(self.stop)


    def __del__(self):
        """Destructor to ensure container is killed when object is garbage collected."""
        self.stop()
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass

    def start(self):
        """Starts a fresh Docker container with Jupyter."""
        logging.info("🐳 Starting Sandbox Container...")

        container_name = f"sandbox_{self.port}"

        try:
            old_container = self.client.containers.get(container_name)
            logging.debug(f"🧹 Found orphan container '{container_name}'. Removing it...")
            old_container.remove(force=True)
        except ContainerNotFound:
            pass
        except Exception as e:
            logging.warning(f"⚠️ Warning: Could not cleanup old container: {e}")


        try:
            self.container = self.client.containers.run(
                self.image_name,
                name=container_name,
                detach=True,
                volumes={self.mount_volume: {'bind': '/app/container', 'mode': 'rw'}} if self.mount_volume else None,
                ports={'8888/tcp': self.port},
                remove=True,
                command=f"jupyter server --allow-root --ServerApp.root_dir=/app/container --ServerApp.ip=0.0.0.0 --ServerApp.websocket_ping_interval=0 --ServerApp.port=8888 --ServerApp.allow_origin='*' --ServerApp.token='{self.token}' --ServerApp.open_browser=False"
            )
            
            self._wait_for_server()
            
            self._start_kernel()
            logging.info("✅ Sandbox Ready.")
            
        except Exception as e:
            logging.error(f"❌ Error starting sandbox: {e}")
            self.stop()
            raise e

    def run_terminal(self, command):
        """
        Executes a shell command inside the container (bypassing Jupyter).
        Returns the combined stdout/stderr string.
        """
        if not self.container:
            raise Exception("Sandbox not started.")

        logging.debug(f"💻 Executing Shell Command: {command}")
        
        # execution_result returns a tuple: (exit_code, output_bytes)
        exit_code, output = self.container.exec_run(
            command, 
            user="root" # Run as root to allow apt-get/installation
        )
        
        return output.decode("utf-8")
    
    def _on_ws_message(self, ws, message):
        """Callback for WebSocketApp running in background thread."""
        try:
            data = json.loads(message)
            parent_msg_id = data.get('parent_header', {}).get('msg_id')

            with self._queue_lock:
                if parent_msg_id and parent_msg_id in self.execution_queues:
                    self.execution_queues[parent_msg_id].put(data)
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")

    def interrupt_kernel(self):
        """Send interrupt signal to the Jupyter kernel."""
        if not self.kernel_id:
            return
            
        try:
            url = f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt"
            requests.post(url, headers={"Authorization": f"Token {self.token}"})
            logging.info("🛑 Kernel interrupt sent")
        except Exception as e:
            logging.warning(f"⚠️ Failed to interrupt kernel: {e}")

    def clean_output(self, outputs):
        """
        Consolidates stream outputs and handles carriage returns (\r) 
        to reduce verbose progress bars (like tqdm/PyMC/Rich).
        """
        if not outputs:
            return []

        stdout_text = []
        stderr_text = []
        non_stream_outputs = []
        
        # Track display_data messages - Rich progress bars send many of these
        # We only want to keep the LAST display_data for progress-like outputs
        display_data_msgs = []
        
        for msg in outputs:
            if msg['msg_type'] == 'stream':
                content = msg['content']
                if content['name'] == 'stdout':
                    stdout_text.append(content['text'])
                else:
                    stderr_text.append(content['text'])
            elif msg['msg_type'] == 'display_data':
                display_data_msgs.append(msg)
            else:
                non_stream_outputs.append(msg)

        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        def clean_stream_text(text):
            """Process a merged stream text to handle \r and ANSI codes."""
            # Remove ANSI escape codes
            text = ansi_escape.sub('', text)
            
            # Simulate a terminal line buffer to handle \r correctly
            # Split by \n first to preserve actual newlines
            lines = text.split('\n')
            final_lines = []
            
            for line in lines:
                if '\r' in line:
                    # Simulate carriage return: only keep what's visible after all \r processing
                    # Each \r moves cursor to start of line, subsequent text overwrites
                    segments = line.split('\r')
                    # The last non-empty segment is what would be displayed
                    visible = ''
                    for segment in segments:
                        if segment:
                            # This segment overwrites from the beginning
                            visible = segment
                    if visible.strip():  # Only add non-empty lines
                        final_lines.append(visible)
                elif line.strip():  # Only add non-empty lines
                    final_lines.append(line)
            
            return '\n'.join(final_lines)
        
        def is_progress_display(msg):
            """PYMC SPECIFIC FUNCTION: Check if a display_data message looks like a progress bar."""
            data = msg.get('content', {}).get('data', {})
            text = data.get('text/plain', '')
            progress_indicators = ['Progress', 'Draws', 'Divergences', 'Step size', 
                                   'Sampling', 'it/s', 'draws/s', '━', '╸', '╺', 
                                   'Output()', 'Elapsed', 'Remaining']
            return any(indicator in text for indicator in progress_indicators)
        
        cleaned_outputs = []
        
        if stdout_text:
            merged_stdout = ''.join(stdout_text)
            cleaned_stdout = clean_stream_text(merged_stdout)
            if cleaned_stdout.strip():
                cleaned_outputs.append({
                    'msg_type': 'stream',
                    'content': {'name': 'stdout', 'text': cleaned_stdout}
                })
        
        if stderr_text:
            merged_stderr = ''.join(stderr_text)
            cleaned_stderr = clean_stream_text(merged_stderr)
            if cleaned_stderr.strip():
                cleaned_outputs.append({
                    'msg_type': 'stream',
                    'content': {'name': 'stderr', 'text': cleaned_stderr}
                })
        
        # Filter display_data: remove progress-like displays entirely, keep others
        last_progress_msg = None
        last_progress_idx = 0
        idx = 0
        for msg in display_data_msgs:
            if not is_progress_display(msg):
                idx += 1
                cleaned_outputs.append(msg)
            else:
                last_progress_idx = idx
                last_progress_msg = msg

        if last_progress_msg:
            cleaned_outputs.insert(last_progress_idx, last_progress_msg)
        
        cleaned_outputs.extend(non_stream_outputs)
        
        return cleaned_outputs
    
    def _execute(
        self,
        code: str,
        cancel_event: Optional[threading.Event] = None,
        poll_interval: float = 0.5,
        timeout: float = 60.0
    ) -> ExecutionResult:
        """
        Core execution logic - used by both run() and run_async().
        
        Args:
            code: Python code to execute
            cancel_event: If set, execution will be interrupted
            poll_interval: How often to check for cancellation (seconds)
            timeout: Timeout for waiting on messages when no cancel_event
        
        Returns:
            ExecutionResult with status and result/error
        """
        if not self.ws_app:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error="Sandbox not started or connection lost."
            )

        msg_id = uuid.uuid4().hex
        execution_queue = queue.Queue()
        
        with self._queue_lock:
            self.execution_queues[msg_id] = execution_queue
        
        message = {
            "header": {
                "msg_id": msg_id,
                "username": "agent",
                "session": uuid.uuid4().hex,
                "msg_type": "execute_request",
                "version": "5.3"
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False
            }
        }

        collected_outputs = []
        
        try:
            self.ws_app.send(json.dumps(message))

            while True:
                # Check for cancellation
                if cancel_event is not None and cancel_event.is_set():
                    self.interrupt_kernel()
                    # Drain remaining messages briefly
                    drain_deadline = time.time() + 2.0
                    while time.time() < drain_deadline:
                        try:
                            response = execution_queue.get(timeout=0.1)
                            if response['msg_type'] == 'status' and \
                               response['content'].get('execution_state') == 'idle':
                                break
                        except queue.Empty:
                            continue
                    
                    return ExecutionResult(
                        status=ExecutionStatus.CANCELLED,
                        result=SandboxResult(self.clean_output(collected_outputs)) if collected_outputs else None
                    )
                
                # Wait for next message
                try:
                    wait_time = poll_interval if cancel_event else timeout
                    response = execution_queue.get(timeout=wait_time)
                except queue.Empty:
                    if cancel_event:
                        continue  # Check cancellation again
                    else:
                        return ExecutionResult(
                            status=ExecutionStatus.ERROR,
                            error="Execution timed out waiting for response"
                        )
                
                msg_type = response['msg_type']
                content = response['content']

                # Check for completion
                if msg_type == 'status' and content.get('execution_state') == 'idle':
                    break
                
                # Collect outputs
                if msg_type in ('stream', 'execute_result', 'display_data', 'error'):
                    collected_outputs.append(response)

            # Check if any outputs contain errors
            has_error = any(msg['msg_type'] == 'error' for msg in collected_outputs)
            
            if has_error:
                # Extract error message from error output
                error_msg = None
                for msg in collected_outputs:
                    if msg['msg_type'] == 'error':
                        error_msg = msg['content'].get('evalue', 'Unknown error')
                        break
                
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    result=SandboxResult(self.clean_output(collected_outputs)),
                    error=error_msg
                )
            
            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                result=SandboxResult(self.clean_output(collected_outputs))
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )
        finally:
            with self._queue_lock:
                self.execution_queues.pop(msg_id, None)

    def run(self, code):
        """Sends code to the container and waits for the result."""
        return self._execute(code, cancel_event=None)
    
    async def run_async(
        self,
        code: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ExecutionResult:
        """Async execution with cancellation support."""
        thread_cancel = threading.Event()
    
        async def watch_cancel():
            if cancel_event:
                await cancel_event.wait()
                thread_cancel.set()

        cancel_watcher = asyncio.create_task(watch_cancel())
        
        try:
            result = await asyncio.to_thread(
                self._execute,
                code,
                thread_cancel,
                0.5,  # poll_interval
                30.0  # timeout
            )
            return result
        except asyncio.CancelledError:
            thread_cancel.set()
            raise
        finally:
            cancel_watcher.cancel()

    def restart(self, cleanup_lambda=None):
        """Kills the current container and starts a fresh one."""
        logging.info("♻️ Restarting Sandbox (Wiping state)...")
        self.stop()

        if cleanup_lambda:
            cleanup_lambda()

        self.start()

    def stop(self):
        """Cleanup: Kill and remove the container."""
        if self.ws_app:
            self.ws_app.close()
            self.ws_app = None
        if self.container:
            try:
                self.container.kill()
                self.container.remove()
            except Exception as e:
                logging.debug(f"Error removing the container: {e}")
                pass

            self.container = None
        logging.info("🛑 Sandbox Destroyed.")

    def _wait_for_server(self):
        """Polls the local port until Jupyter is responding."""
        retries = 20
        while retries > 0:
            try:
                requests.get(f"{self.base_url}/api", headers={"Authorization": f"Token {self.token}"})
                return
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
                retries -= 1
        raise Exception("Jupyter Server failed to start.")

    def _start_kernel(self):
        """Uses REST API to start a kernel and WebSocket to connect."""
        response = requests.post(
            f"{self.base_url}/api/kernels",
            headers={"Authorization": f"Token {self.token}"}
        )
        self.kernel_id = response.json()["id"]
        
        ws_url = f"ws://127.0.0.1:{self.port}/api/kernels/{self.kernel_id}/channels?token={self.token}"
        
        self.ws_open_event.clear()
        
        self.ws_app = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_ws_message,
            on_open=lambda ws: self.ws_open_event.set()
        )
        
        self.ws_thread = threading.Thread(
            target=self.ws_app.run_forever, 
            kwargs={'ping_interval': 30, 'ping_timeout': 10}
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        if not self.ws_open_event.wait(timeout=10):
            raise Exception("Failed to establish WebSocket connection to Kernel.")