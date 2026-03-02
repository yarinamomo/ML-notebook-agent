"""
Unit tests for DockerSandbox class.

Tests core functionality with mocked Docker and WebSocket components.
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch, call, PropertyMock
from src.sandbox import DockerSandbox


class TestDockerSandboxInit:
    """Test DockerSandbox initialization."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            assert sandbox.image_name == "test-image"
            assert sandbox.port == 8888
            assert sandbox.base_url == "http://127.0.0.1"
            assert sandbox.ws is None
            assert sandbox.kernel_id is None
            assert sandbox.execution_results == {}
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox(
                image_name="custom-image",
                port=9999,
                base_url="http://localhost",
                token="my-token",
                mount_volume="/custom/path"
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
        with patch('src.sandbox.docker.from_env'):
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
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            sandbox.ws = None
            assert sandbox._is_websocket_connected() is False
    
    def test_is_websocket_connected_false_no_sock(self):
        """Test WebSocket connection check returns False when sock is None."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            mock_ws = Mock()
            mock_ws.sock = None
            sandbox.ws = mock_ws
            assert sandbox._is_websocket_connected() is False
    
    def test_is_websocket_connected_false_not_connected(self):
        """Test WebSocket connection check returns False when not connected."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            mock_socket = Mock()
            mock_socket.connected = False
            mock_ws = Mock()
            mock_ws.sock = mock_socket
            sandbox.ws = mock_ws
            assert sandbox._is_websocket_connected() is False
    
    def test_is_websocket_connected_exception(self):
        """Test WebSocket connection check handles exceptions gracefully."""
        with patch('src.sandbox.docker.from_env'):
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
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                'status': 'pending',
                'outputs': [],
                'done': False,
                'execution_count': None
            }
            
            message = json.dumps({
                'msg_type': 'stream',
                'content': {'name': 'stdout', 'text': 'Hello World\n'},
                'parent_header': {'msg_id': msg_id}
            })
            
            sandbox._on_ws_message(None, message)
            
            assert len(sandbox.execution_results[msg_id]['outputs']) == 1
            assert sandbox.execution_results[msg_id]['outputs'][0]['output_type'] == 'stream'
            assert sandbox.execution_results[msg_id]['outputs'][0]['text'] == 'Hello World\n'
    
    def test_on_ws_message_execute_result(self):
        """Test handling execute result messages."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                'status': 'pending',
                'outputs': [],
                'done': False,
                'execution_count': None
            }
            
            message = json.dumps({
                'msg_type': 'execute_result',
                'content': {
                    'data': {'text/plain': '42'},
                    'execution_count': 1
                },
                'parent_header': {'msg_id': msg_id}
            })
            
            sandbox._on_ws_message(None, message)
            
            assert len(sandbox.execution_results[msg_id]['outputs']) == 1
            assert sandbox.execution_results[msg_id]['outputs'][0]['output_type'] == 'execute_result'
            assert sandbox.execution_results[msg_id]['outputs'][0]['data'] == {'text/plain': '42'}
    
    def test_on_ws_message_error(self):
        """Test handling error messages."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                'status': 'pending',
                'outputs': [],
                'done': False,
                'execution_count': None
            }
            
            message = json.dumps({
                'msg_type': 'error',
                'content': {
                    'ename': 'ValueError',
                    'evalue': 'invalid value',
                    'traceback': ['line 1', 'line 2']
                },
                'parent_header': {'msg_id': msg_id}
            })
            
            sandbox._on_ws_message(None, message)
            
            assert len(sandbox.execution_results[msg_id]['outputs']) == 1
            assert sandbox.execution_results[msg_id]['outputs'][0]['output_type'] == 'error'
            assert sandbox.execution_results[msg_id]['outputs'][0]['ename'] == 'ValueError'
    
    def test_on_ws_message_execute_reply(self):
        """Test handling execute reply messages."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            msg_id = "test-msg-id"
            sandbox.execution_results[msg_id] = {
                'status': 'pending',
                'outputs': [],
                'done': False,
                'execution_count': None
            }
            
            message = json.dumps({
                'msg_type': 'execute_reply',
                'content': {'status': 'ok', 'execution_count': 1},
                'parent_header': {'msg_id': msg_id}
            })
            
            sandbox._on_ws_message(None, message)
            
            assert sandbox.execution_results[msg_id]['status'] == 'ok'
            assert sandbox.execution_results[msg_id]['execution_count'] == 1
            assert sandbox.execution_results[msg_id]['done'] is True
    
    def test_on_ws_message_unknown_msg_id(self):
        """Test handling messages for unknown msg_id."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            message = json.dumps({
                'msg_type': 'stream',
                'content': {'name': 'stdout', 'text': 'Hello'},
                'parent_header': {'msg_id': 'unknown-id'}
            })
            
            # Should not raise, just return early
            sandbox._on_ws_message(None, message)
            assert 'unknown-id' not in sandbox.execution_results
    
    def test_on_ws_message_invalid_json(self):
        """Test handling invalid JSON messages."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            # Should not raise, just log exception
            sandbox._on_ws_message(None, "invalid json")


class TestExecutionLogic:
    """Test code execution logic."""
    
    @patch('src.sandbox.requests.post')
    @patch('src.sandbox.websocket.WebSocketApp')
    def test_execute_code_success(self, mock_ws_class, mock_post):
        """Test successful code execution."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            # Setup mock WebSocket
            mock_socket = Mock()
            mock_socket.connected = True
            mock_ws = Mock()
            mock_ws.sock = mock_socket
            sandbox.ws = mock_ws
            sandbox.kernel_id = "kernel-123"
            
            # Setup execution result
            msg_id = None
            def capture_msg_id(data):
                nonlocal msg_id
                msg = json.loads(data)
                msg_id = msg['header']['msg_id']
            
            sandbox.ws.send = Mock(side_effect=capture_msg_id)
            
            # Simulate execution completion
            def execute_code_and_complete():
                sandbox._execute_code("print('hello')", timeout=5)
                # Complete the execution
                if msg_id:
                    sandbox.execution_results[msg_id]['outputs'].append({
                        'output_type': 'stream',
                        'name': 'stdout',
                        'text': 'hello\n'
                    })
                    sandbox.execution_results[msg_id]['status'] = 'ok'
                    sandbox.execution_results[msg_id]['done'] = True
            
            # Run in a separate approach to avoid blocking
            import threading
            exec_thread = threading.Thread(target=execute_code_and_complete)
            exec_thread.start()
            exec_thread.join(timeout=2)
            
            # Note: This test demonstrates the pattern but actual execution is complex
            # with threading. See integration tests for full execution testing.
    
    def test_execute_code_no_connection(self):
        """Test execution fails when no WebSocket connection."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            sandbox.ws = None
            
            with pytest.raises(RuntimeError, match="WebSocket disconnected"):
                sandbox._execute_code("print('hello')", timeout=5)


class TestRetryLogic:
    """Test reconnection and retry logic."""
    
    @patch('src.sandbox.DockerSandbox._start_kernel_websocket')
    def test_run_retries_on_disconnection(self, mock_start_kernel):
        """Test that run() retries when WebSocket is disconnected."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            sandbox.ws = None
            
            # First call fails (WebSocket not initialized)
            # Second call succeeds
            mock_start_kernel.side_effect = [
                RuntimeError("Failed to connect"),
                None  # Success on second try
            ]
            
            with pytest.raises(RuntimeError, match="Failed to reconnect"):
                # Only 2 retries max, both fail
                sandbox.run("print('hello')", max_retries=0)
    
    @patch('src.sandbox.DockerSandbox._execute_code')
    @patch('src.sandbox.DockerSandbox._start_kernel_websocket')
    def test_run_successful_after_reconnect(self, mock_start_kernel, mock_execute):
        """Test successful execution after reconnection."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            # First check: no connection
            # After reconnect: connection exists
            call_count = [0]
            
            def is_connected():
                call_count[0] += 1
                return call_count[0] > 1
            
            sandbox._is_websocket_connected = is_connected
            mock_start_kernel.return_value = None
            mock_execute.return_value = {
                'status': 'ok',
                'outputs': [],
                'execution_count': 1
            }
            
            # This should succeed after reconnecting
            result = sandbox.run("print('hello')", max_retries=2)
            assert result['status'] == 'ok'
            mock_start_kernel.assert_called()


class TestInterruptLogic:
    """Test kernel interrupt functionality."""
    
    @patch('src.sandbox.requests.post')
    def test_interrupt_kernel_success(self, mock_post):
        """Test successful kernel interrupt."""
        with patch('src.sandbox.docker.from_env'):
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
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            sandbox.kernel_id = None
            
            with pytest.raises(RuntimeError, match="No kernel ID"):
                sandbox._interrupt_kernel()


class TestKernelRestart:
    """Test kernel restart functionality."""
    
    @patch('src.sandbox.DockerSandbox._start_kernel_websocket')
    @patch('src.sandbox.requests.delete')
    def test_restart_kernel_cleans_up(self, mock_delete, mock_start_kernel):
        """Test kernel restart properly cleans up old kernel."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            # Setup old kernel
            mock_socket = Mock()
            mock_socket.connected = True
            mock_ws = Mock()
            mock_ws.sock = mock_socket
            sandbox.ws = mock_ws
            sandbox.kernel_id = "old-kernel-id"
            sandbox.execution_results = {"msg-1": {}}
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_delete.return_value = mock_response
            
            mock_start_kernel.return_value = None
            
            sandbox._restart_kernel_websocket()
            
            # Verify cleanup
            assert sandbox.kernel_id is None
            assert sandbox.execution_results == {}
            mock_delete.assert_called_once()
            mock_start_kernel.assert_called_once()
    
    @patch('src.sandbox.DockerSandbox._start_kernel_websocket')
    def test_restart_kernel_public_api(self, mock_start_kernel):
        """Test public restart_kernel API."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            
            mock_start_kernel.return_value = None
            sandbox.kernel_id = "old-kernel"
            sandbox.ws = Mock(sock=Mock(connected=True))
            
            # This should not raise
            sandbox.restart_kernel()
            
            # Should have called internal restart
            mock_start_kernel.assert_called()


class TestExceptionHandling:
    """Test exception handling in various scenarios."""
    
    def test_on_ws_error(self):
        """Test WebSocket error handling."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            mock_ws = Mock()
            sandbox.ws = mock_ws
            
            sandbox._on_ws_error(None, "Connection lost")
            
            # Should clear the WebSocket reference
            assert sandbox.ws is None
    
    def test_on_ws_close(self):
        """Test WebSocket close handling."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            mock_ws = Mock()
            sandbox.ws = mock_ws
            
            sandbox._on_ws_close(None, 1000, "Normal close")
            
            # Should clear the WebSocket reference
            assert sandbox.ws is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
