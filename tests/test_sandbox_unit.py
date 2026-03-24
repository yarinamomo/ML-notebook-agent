"""
Unit tests for DockerSandbox class.

Tests core functionality with mocked Docker and WebSocket components.
"""

import json
from typing import cast
from unittest.mock import Mock, PropertyMock, patch

import pytest

from src.sandbox import DockerSandbox
from src.ui_agent import EnvironmentUnavailable
from src.utils.nb_types import (
    CellExecutionResult,
    ErrorOutput,
    ExecuteResultOutput,
    StreamOutput,
)
from src.utils.retry_sandbox import check_websocket_connected, retry_on_failure


class TestDockerSandboxInit:
    """Test DockerSandbox initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            assert sandbox.image_name == "test-image"
            assert sandbox.port == 8888
            assert sandbox.base_url == "http://127.0.0.1"
            assert sandbox.ws is None
            assert sandbox.kernel_id is None
            assert sandbox.execution_results == {}

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox(
                image_name="custom-image",
                port=9999,
                base_url="http://localhost",
                token="my-token",
                mount_volume="/custom/path",
            )
            assert sandbox.image_name == "custom-image"
            assert sandbox.port == 9999
            assert sandbox.base_url == "http://localhost"
            assert sandbox.token == "my-token"
            assert sandbox.mount_volume == "/custom/path"


class TestWebSocketConnection:
    """Test WebSocket connection checks."""

    def test_is_websocket_connected_true(self):
        """Test WebSocket connection check returns True."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            # Mock a connected WebSocket
            mock_socket = Mock()
            mock_socket.connected = True
            mock_ws = Mock()
            mock_ws.sock = mock_socket
            sandbox.ws = mock_ws

            assert sandbox._is_websocket_connected() is True

    def test_is_websocket_connected_false_no_ws(self):
        """Test WebSocket connection check returns False when ws is None."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.ws = None
            assert sandbox._is_websocket_connected() is False

    def test_is_websocket_connected_false_no_sock(self):
        """Test WebSocket connection check returns False when sock is None."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            mock_ws = Mock()
            mock_ws.sock = None
            sandbox.ws = mock_ws
            assert sandbox._is_websocket_connected() is False

    def test_is_websocket_connected_false_not_connected(self):
        """Test WebSocket connection check returns False when not connected."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            mock_socket = Mock()
            mock_socket.connected = False
            mock_ws = Mock()
            mock_ws.sock = mock_socket
            sandbox.ws = mock_ws
            assert sandbox._is_websocket_connected() is False

    def test_is_websocket_connected_exception(self):
        """Test WebSocket connection check handles exceptions gracefully."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            mock_ws = Mock()
            # Make accessing .sock raise an exception
            type(mock_ws).sock = PropertyMock(side_effect=Exception("Connection error"))
            sandbox.ws = mock_ws
            assert sandbox._is_websocket_connected() is False


class TestMessageHandling:
    """Test WebSocket message handling."""

    def test_on_ws_message_stream(self):
        """Test handling stream messages."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                "status": "pending",
                "outputs": [],
                "done": False,
                "execution_count": None,
            }

            message = json.dumps(
                {
                    "msg_type": "stream",
                    "content": {"name": "stdout", "text": "Hello World\n"},
                    "parent_header": {"msg_id": msg_id},
                }
            )

            sandbox._on_ws_message(None, message)

            assert len(sandbox.execution_results[msg_id]["outputs"]) == 1
            output = cast(StreamOutput, sandbox.execution_results[msg_id]["outputs"][0])
            assert output["output_type"] == "stream"
            assert output["text"] == "Hello World\n"

    def test_on_ws_message_execute_result(self):
        """Test handling execute result messages."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                "status": "pending",
                "outputs": [],
                "done": False,
                "execution_count": None,
            }

            message = json.dumps(
                {
                    "msg_type": "execute_result",
                    "content": {"data": {"text/plain": "42"}, "execution_count": 1},
                    "parent_header": {"msg_id": msg_id},
                }
            )

            sandbox._on_ws_message(None, message)

            assert len(sandbox.execution_results[msg_id]["outputs"]) == 1
            output = cast(
                ExecuteResultOutput, sandbox.execution_results[msg_id]["outputs"][0]
            )
            assert output["output_type"] == "execute_result"
            assert output["data"] == {"text/plain": "42"}

    def test_on_ws_message_error(self):
        """Test handling error messages."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                "status": "pending",
                "outputs": [],
                "done": False,
                "execution_count": None,
            }

            message = json.dumps(
                {
                    "msg_type": "error",
                    "content": {
                        "ename": "ValueError",
                        "evalue": "invalid value",
                        "traceback": ["line 1", "line 2"],
                    },
                    "parent_header": {"msg_id": msg_id},
                }
            )

            sandbox._on_ws_message(None, message)

            assert len(sandbox.execution_results[msg_id]["outputs"]) == 1
            output = cast(ErrorOutput, sandbox.execution_results[msg_id]["outputs"][0])
            assert output["output_type"] == "error"
            assert output["ename"] == "ValueError"

    def test_on_ws_message_execute_reply(self):
        """Test handling execute reply messages."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                "status": "pending",
                "outputs": [],
                "done": False,
                "execution_count": None,
            }

            message = json.dumps(
                {
                    "msg_type": "execute_reply",
                    "content": {"status": "ok", "execution_count": 1},
                    "parent_header": {"msg_id": msg_id},
                }
            )

            sandbox._on_ws_message(None, message)

            assert sandbox.execution_results[msg_id]["status"] == "ok"
            assert sandbox.execution_results[msg_id]["execution_count"] == 1
            assert sandbox.execution_results[msg_id]["done"] is True

    def test_on_ws_message_unknown_msg_id(self):
        """Test handling messages for unknown msg_id."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            message = json.dumps(
                {
                    "msg_type": "stream",
                    "content": {"name": "stdout", "text": "Hello"},
                    "parent_header": {"msg_id": "unknown-id"},
                }
            )

            # Should not raise, just return early
            sandbox._on_ws_message(None, message)
            assert "unknown-id" not in sandbox.execution_results

    def test_on_ws_message_invalid_json(self):
        """Test handling invalid JSON messages."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            # Should not raise, just log exception
            sandbox._on_ws_message(None, "invalid json")


class TestExecutionLogic:
    """Test code execution logic."""

    @patch("src.sandbox.DockerSandbox.restart_kernel")
    def test_run_no_connection_raises_error(self, mock_restart):
        """Test execution fails when no WebSocket connection."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.ws = None

            # Mock restart_kernel to fail
            mock_restart.side_effect = RuntimeError("Failed to restart")

            with pytest.raises((RuntimeError, Exception)):
                sandbox.run("print('hello')", timeout=5)


class TestRetryLogic:
    """Test reconnection and retry logic."""

    @patch("src.sandbox.DockerSandbox.restart_kernel")
    def test_run_retries_on_disconnection(self, mock_restart):
        """Test that run() retries when WebSocket is disconnected."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.ws = None

            # Restart keeps failing
            mock_restart.side_effect = RuntimeError("Failed to restart")

            with pytest.raises((RuntimeError, Exception)):
                sandbox.run("print('hello')")

    @patch("src.sandbox.DockerSandbox.restart_kernel")
    def test_run_successful_after_reconnect(self, mock_restart):
        """Test successful execution after reconnection."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            # First check: no connection
            # After reconnect: connection exists
            call_count = [0]

            def is_connected():
                call_count[0] += 1
                return call_count[0] > 1

            sandbox._is_websocket_connected = is_connected

            # Setup mock WebSocket for actual execution
            mock_socket = Mock()
            mock_socket.connected = True
            mock_ws = Mock()
            mock_ws.sock = mock_socket

            def setup_ws():
                sandbox.ws = mock_ws
                sandbox.kernel_id = "test-kernel"

            mock_restart.side_effect = setup_ws

            # Mock the send to complete execution immediately
            def complete_execution(msg_str):
                msg = json.loads(msg_str)
                msg_id = msg["header"]["msg_id"]
                if msg_id in sandbox.execution_results:
                    sandbox.execution_results[msg_id]["status"] = "ok"
                    sandbox.execution_results[msg_id]["done"] = True

            mock_ws.send = Mock(side_effect=complete_execution)

            # This should succeed after reconnecting
            result = sandbox.run("print('hello')")
            assert result["status"] == "ok"
            mock_restart.assert_called()


class TestDecoratorBehavior:
    """Test decorator behavior."""

    def test_check_websocket_connected_reconnects_if_needed(self):
        """check_websocket_connected should restart kernel if disconnected."""
        mock_sandbox = Mock()
        mock_sandbox._is_websocket_connected.return_value = False
        mock_sandbox.restart_kernel = Mock()

        @check_websocket_connected()
        def fake_operation(self):
            return "success"

        result = fake_operation(mock_sandbox)

        assert result == "success"
        mock_sandbox.restart_kernel.assert_called_once()

    def test_retry_on_failure_raises_environment_unavailable(self):
        """retry_on_failure should raise EnvironmentUnavailable after max retries."""
        mock_sandbox = Mock()

        call_count = {"count": 0}

        @retry_on_failure(max_retries=1)
        def always_fails(self):
            call_count["count"] += 1
            raise RuntimeError("persistent failure")

        with pytest.raises(EnvironmentUnavailable):
            always_fails(mock_sandbox)

        # Should have tried initial attempt + 1 retry = 2 times total
        assert call_count["count"] == 2

    def test_retry_on_failure_succeeds_on_retry(self):
        """retry_on_failure should succeed if operation succeeds on retry."""
        mock_sandbox = Mock()

        call_count = {"count": 0}

        @retry_on_failure(max_retries=2)
        def flaky_operation(self):
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise RuntimeError("temporary failure")
            return "ok"

        result = flaky_operation(mock_sandbox)

        assert result == "ok"
        assert call_count["count"] == 2


class TestInterruptLogic:
    """Test kernel interrupt functionality."""

    @patch("src.sandbox.requests.post")
    def test_interrupt_kernel_success(self, mock_post):
        """Test successful kernel interrupt."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.kernel_id = "kernel-123"

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            sandbox._interrupt_kernel()

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "interrupt" in call_args[0][0]
            assert "kernel-123" in call_args[0][0]

    def test_interrupt_kernel_no_kernel_id(self):
        """Test interrupt fails when no kernel_id."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.kernel_id = None

            with pytest.raises(RuntimeError, match="No kernel ID"):
                sandbox._interrupt_kernel()


class TestKernelRestart:
    """Test kernel restart functionality."""

    @patch("src.sandbox.DockerSandbox._start_kernel")
    @patch("src.sandbox.DockerSandbox._start_websocket")
    @patch("src.sandbox.requests.delete")
    def test_restart_kernel_cleans_up(
        self, mock_delete, mock_start_ws, mock_start_kernel
    ):
        """Test kernel restart properly cleans up old kernel."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            # Setup old kernel
            mock_socket = Mock()
            mock_socket.connected = True
            mock_ws = Mock()
            mock_ws.sock = mock_socket
            sandbox.ws = mock_ws
            sandbox.kernel_id = "old-kernel-id"
            sandbox.execution_results = {
                "msg-1": cast(
                    CellExecutionResult,
                    {"status": "ok", "done": True, "execution_count": 1, "outputs": []},
                )
            }

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_delete.return_value = mock_response

            mock_start_kernel.return_value = None
            mock_start_ws.return_value = None

            sandbox.restart_kernel()

            # Verify cleanup
            assert sandbox.execution_results == {}
            mock_delete.assert_called_once()
            mock_start_kernel.assert_called_once()
            mock_start_ws.assert_called_once()

    @patch("src.sandbox.DockerSandbox._start_kernel")
    @patch("src.sandbox.DockerSandbox._start_websocket")
    @patch("src.sandbox.DockerSandbox._stop_websocket")
    @patch("src.sandbox.DockerSandbox._stop_kernel")
    def test_restart_kernel_public_api(
        self, mock_stop_kernel, mock_stop_ws, mock_start_ws, mock_start_kernel
    ):
        """Test public restart_kernel API."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            mock_start_kernel.return_value = None
            mock_start_ws.return_value = None
            sandbox.kernel_id = "old-kernel"
            sandbox.ws = Mock(sock=Mock(connected=True))

            # This should not raise
            sandbox.restart_kernel()

            # Should have called internal methods
            mock_stop_ws.assert_called_once()
            mock_stop_kernel.assert_called_once()
            mock_start_kernel.assert_called_once()
            mock_start_ws.assert_called_once()


class TestKernelAndWebSocketManagement:
    """Test individual kernel and websocket management methods."""

    @patch("src.sandbox.requests.post")
    def test_start_kernel(self, mock_post):
        """Test starting a kernel."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            mock_response = Mock()
            mock_response.json.return_value = {"id": "new-kernel-123"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            sandbox._start_kernel()

            assert sandbox.kernel_id == "new-kernel-123"
            mock_post.assert_called_once()

    @patch("src.sandbox.websocket.WebSocketApp")
    @patch("src.sandbox.threading.Thread")
    def test_start_websocket(self, mock_thread, mock_ws_class):
        """Test starting websocket connection."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.kernel_id = "test-kernel-id"

            # Setup mock WebSocket
            mock_socket = Mock()
            mock_socket.connected = True
            mock_ws = Mock()
            mock_ws.sock = mock_socket
            mock_ws_class.return_value = mock_ws

            # Setup mock thread
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            sandbox._start_websocket()

            assert sandbox.ws == mock_ws
            mock_ws_class.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_start_websocket_without_kernel_id(self):
        """Test starting websocket without kernel_id raises error."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.kernel_id = None

            with pytest.raises(RuntimeError, match="No kernel ID"):
                sandbox._start_websocket()

    @patch("src.sandbox.requests.delete")
    def test_stop_kernel(self, mock_delete):
        """Test stopping a kernel."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            sandbox.kernel_id = "test-kernel-id"

            mock_response = Mock()
            mock_delete.return_value = mock_response

            sandbox._stop_kernel()

            assert sandbox.kernel_id is None
            mock_delete.assert_called_once()

    def test_stop_websocket(self):
        """Test stopping websocket connection."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")

            # Setup mock WebSocket
            mock_ws = Mock()
            sandbox.ws = mock_ws

            # Setup mock thread
            mock_thread = Mock()
            mock_thread.is_alive.return_value = False
            sandbox.ws_thread = mock_thread

            sandbox._stop_websocket()

            assert sandbox.ws is None
            mock_ws.close.assert_called_once()


class TestExceptionHandling:
    """Test exception handling in various scenarios."""

    def test_on_ws_error(self):
        """Test WebSocket error handling."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            mock_ws = Mock()
            sandbox.ws = mock_ws

            sandbox._on_ws_error(None, "Connection lost")

            # Should clear the WebSocket reference
            assert sandbox.ws is None

    def test_on_ws_close(self):
        """Test WebSocket close handling."""
        with patch("src.sandbox.docker.from_env"):
            sandbox = DockerSandbox("test-image")
            mock_ws = Mock()
            sandbox.ws = mock_ws

            sandbox._on_ws_close(None, 1000, "Normal close")

            # Should clear the WebSocket reference
            assert sandbox.ws is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
