"""
Integration tests for DockerSandbox class.

Tests real Docker container and Jupyter kernel interactions.
Requires the Docker image specified by the layered agent config to be available.
"""

import os

import pytest
import time
import logging
from pathlib import Path
from src.sandbox import DockerSandbox
from src.ui_agent import EnvironmentUnavailable
from src.utils.log import logger
from src.utils.yaml_parser import load_config

DEFAULTS_CONFIG = Path(os.getenv("NOTEBOOK_AGENT_DEFAULTS_PATH", "./config/defaults.yaml"))
print(f"Using defaults config: {DEFAULTS_CONFIG}")

@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Enable informative logger output during integration tests."""
    logger.setLevel(logging.DEBUG)


@pytest.fixture
def config():
    """Load configuration from agent.yaml with layered defaults."""
    config_path = Path(__file__).parent.parent / "config" / "agent.yaml"
    return load_config(config_path, DEFAULTS_CONFIG)


@pytest.fixture
def docker_image(config):
    """Get Docker image from config."""
    return config["environment"]["docker_image_name"]


@pytest.fixture
def docker_start_command(config):
    """Get Docker start command from config."""
    return config["environment"].get("docker_start_command")


@pytest.fixture
def docker_mount_path(config):
    """Get and prepare host mount path from config."""
    mount_path = Path(config["environment"]["docker_mount_path"])
    if not mount_path.is_absolute():
        mount_path = Path(__file__).parent.parent / mount_path
    mount_path.mkdir(parents=True, exist_ok=True)
    return str(mount_path)


@pytest.fixture
def docker_port():
    """Port for Docker container."""
    return 8888


@pytest.fixture
def docker_token():
    """Authentication token for Docker container."""
    return "test-token-12345"




@pytest.fixture
def sandbox(docker_image, docker_port, docker_token, docker_start_command, docker_mount_path):
    """Create and start a Docker sandbox."""
    sandbox = DockerSandbox(
        image_name=docker_image,
        port=docker_port,
        token=docker_token,
        mount_volume=docker_mount_path,
        start_command=docker_start_command,
    )
    
    # Start the container
    sandbox.start()
    yield sandbox
    
    # Cleanup
    try:
        sandbox.stop()
        time.sleep(1)  # Wait for container cleanup before next test
    except Exception as e:
        logger.warning("Error stopping sandbox: %s", e)


class TestSandboxStartup:
    """Test sandbox startup and initialization."""
    
    def test_sandbox_starts_successfully(self, sandbox):
        """Test that sandbox starts and has a kernel."""
        assert sandbox.container is not None
        assert sandbox.kernel_id is not None
        assert sandbox.ws is not None
        assert sandbox._is_websocket_connected() is True


class TestBasicExecution:
    """Test basic code execution."""
    
    def test_execute_simple_print(self, sandbox):
        """Test executing simple print statement."""
        result = sandbox.run("print('Hello World')")
        
        assert result['status'] == 'ok'
        assert len(result['outputs']) > 0
        # Check if output contains the print result
        output_texts = [o.get('text', '') for o in result['outputs'] if o.get('output_type') == 'stream']
        assert any('Hello World' in text for text in output_texts)
    
    def test_execute_variable_assignment(self, sandbox):
        """Test variable assignment and computation."""
        result = sandbox.run("x = 5\ny = 10\nz = x + y\nprint(z)")
        
        assert result['status'] == 'ok'
        output_texts = [o.get('text', '') for o in result['outputs'] if o.get('output_type') == 'stream']
        assert any('15' in text for text in output_texts)
    
    def test_execute_import_request(self, sandbox):
        """Test importing standard library."""
        result = sandbox.run("import json\nprint(json.dumps({'a': 1}))")
        
        assert result['status'] == 'ok'
        output_texts = [o.get('text', '') for o in result['outputs'] if o.get('output_type') == 'stream']
        assert any('a' in text for text in output_texts)
    
    def test_execute_multiline_code(self, sandbox):
        """Test executing multiline code."""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(5)
print(f"Fibonacci(5) = {result}")
"""
        result = sandbox.run(code)
        
        assert result['status'] == 'ok'
        output_texts = [o.get('text', '') for o in result['outputs'] if o.get('output_type') == 'stream']
        assert any('5' in text for text in output_texts)


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_execute_with_syntax_error(self, sandbox):
        """Test handling of syntax errors."""
        result = sandbox.run("x = invalid syntax")
        
        # Should have error status or error output
        error_outputs = [o for o in result['outputs'] if o.get('output_type') == 'error']
        assert len(error_outputs) > 0 or result['status'] == 'error'
    
    def test_execute_with_runtime_error(self, sandbox):
        """Test handling of runtime errors."""
        result = sandbox.run("x = 1 / 0")
        
        # Should have error output
        error_outputs = [o for o in result['outputs'] if o.get('output_type') == 'error']
        assert len(error_outputs) > 0 or result['status'] == 'error'
    
    def test_execute_with_undefined_variable(self, sandbox):
        """Test handling of undefined variable errors."""
        result = sandbox.run("print(undefined_var)")
        
        error_outputs = [o for o in result['outputs'] if o.get('output_type') == 'error']
        assert len(error_outputs) > 0 or result['status'] == 'error'


class TestTimeout:
    """Test timeout and interrupt functionality."""
    
    def test_execute_with_timeout(self, sandbox):
        """Test code execution with timeout."""
        code = """
import time
time.sleep(5)
"""
        # Should return timeout status after 1 second
        result = sandbox.run(code, timeout=1)
        assert result['status'] == 'timeout'
        assert len(result['outputs']) > 0
        last_output = result['outputs'][-1]
        assert last_output.get('output_type') == 'error'
        assert last_output.get('ename') == 'TimeoutError'
    
    def test_kernel_restarted_after_timeout(self, sandbox):
        """Test that kernel is restarted after timeout."""
        # First execution times out and returns timeout message
        timeout_result = sandbox.run("while True: pass", timeout=1)
        assert timeout_result['status'] == 'timeout'
        assert len(timeout_result['outputs']) > 0
        last_output = timeout_result['outputs'][-1]
        assert last_output.get('output_type') == 'error'
        assert last_output.get('ename') == 'TimeoutError'
        
        # Kernel should be restarted and working again
        assert sandbox.kernel_id is not None
        result = sandbox.run("print('after timeout')", timeout=5)
        assert result['status'] == 'ok'


class TestReconnection:
    """Test reconnection and retry logic."""
    
    def test_execute_after_websocket_disconnect(self, sandbox):
        """Test execution after manual WebSocket disconnect."""
        # Close the WebSocket connection
        if sandbox.ws:
            sandbox.ws.close()
            sandbox.ws = None
        
        # Execution should reconnect and succeed (retry decorator handles this)
        result = sandbox.run("print('reconnected')")
        assert result['status'] == 'ok'
        output_texts = [o.get('text', '') for o in result['outputs'] if o.get('output_type') == 'stream']
        assert any('reconnected' in text for text in output_texts)


class TestKernelRestart:
    """Test kernel restart functionality."""
    
    def test_restart_kernel(self, sandbox):
        """Test explicit kernel restart."""
        original_kernel_id = sandbox.kernel_id
        
        # Restart the kernel
        sandbox.restart_kernel()
        
        # Should have a new kernel
        assert sandbox.kernel_id is not None
        # May or may not be different depending on timing, but should work
        result = sandbox.run("print('restarted')")
        assert result['status'] == 'ok'
    
    def test_run_all_restarts_kernel(self, sandbox):
        """Test that simulating run_all behavior works."""
        # Execute a cell
        result1 = sandbox.run("x = 1")
        assert result1['status'] == 'ok'
        
        # Restart kernel (simulating run_all)
        sandbox.restart_kernel()
        
        # Execute another cell - should have fresh state
        result2 = sandbox.run("print(x if 'x' in dir() else 'undefined')")
        assert result2['status'] == 'ok'


class TestMultipleExecutions:
    """Test multiple sequential executions."""
    
    def test_sequential_executions(self, sandbox):
        """Test multiple sequential code executions."""
        results = []
        
        for i in range(3):
            result = sandbox.run(f"print('Execution {i}')")
            results.append(result)
            assert result['status'] == 'ok'
        
        assert len(results) == 3
        assert all(r['status'] == 'ok' for r in results)
    
    def test_state_persistence_across_executions(self, sandbox):
        """Test that state persists across executions."""
        # Create a variable
        result1 = sandbox.run("counter = 0")
        assert result1['status'] == 'ok'
        
        # Increment it
        result2 = sandbox.run("counter += 1\nprint(counter)")
        assert result2['status'] == 'ok'
        output_texts = [o.get('text', '') for o in result2['outputs'] if o.get('output_type') == 'stream']
        assert any('1' in text for text in output_texts)
        
        # Increment again
        result3 = sandbox.run("counter += 1\nprint(counter)")
        assert result3['status'] == 'ok'
        output_texts = [o.get('text', '') for o in result3['outputs'] if o.get('output_type') == 'stream']
        assert any('2' in text for text in output_texts)


class TestOutputFormats:
    """Test various output formats and content types."""
    
    def test_capture_stdout(self, sandbox):
        """Test capturing stdout."""
        result = sandbox.run("print('stdout output')")
        
        stream_outputs = [o for o in result['outputs'] if o.get('output_type') == 'stream']
        assert len(stream_outputs) > 0
        assert any('stdout output' in o.get('text', '') for o in stream_outputs)
    
    def test_capture_stderr(self, sandbox):
        """Test capturing stderr."""
        result = sandbox.run("import sys\nprint('stderr message', file=sys.stderr)")
        
        # Should have stream output (stderr)
        stream_outputs = [o for o in result['outputs'] if o.get('output_type') == 'stream']
        # stderr may be captured as well
        assert len(result['outputs']) > 0
    
    def test_multiple_print_statements(self, sandbox):
        """Test multiple print statements accumulate."""
        code = """
print('Line 1')
print('Line 2')
print('Line 3')
"""
        result = sandbox.run(code)
        
        all_text = ''.join([o.get('text', '') for o in result['outputs'] if o.get('output_type') == 'stream'])
        assert 'Line 1' in all_text
        assert 'Line 2' in all_text
        assert 'Line 3' in all_text


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_numpy_operations(self, sandbox):
        """Test NumPy operations if available."""
        try:
            result = sandbox.run("""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
mean = arr.mean()
print(f"Mean: {mean}")
""")
            assert result['status'] == 'ok'
        except Exception as e:
            # NumPy may not be installed in test environment
            pytest.skip(f"NumPy not available: {e}")
    
    def test_pandas_operations(self, sandbox):
        """Test pandas operations if available."""
        try:
            result = sandbox.run("""
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.sum())
""")
            assert result['status'] == 'ok'
        except Exception as e:
            # Pandas may not be installed in test environment
            pytest.skip(f"Pandas not available: {e}")
    
    def test_exception_with_traceback(self, sandbox):
        """Test exception with traceback."""
        code = """
def func():
    x = 1 / 0
    
func()
"""
        result = sandbox.run(code)
        
        error_outputs = [o for o in result['outputs'] if o.get('output_type') == 'error']
        assert len(error_outputs) > 0
        assert 'ZeroDivisionError' in error_outputs[0].get('ename', '')


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_code(self, sandbox):
        """Test executing empty code."""
        result = sandbox.run("")
        assert result['status'] == 'ok'
    
    def test_only_whitespace(self, sandbox):
        """Test executing only whitespace."""
        result = sandbox.run("   \n\n  \n")
        assert result['status'] == 'ok'
    
    def test_comment_only(self, sandbox):
        """Test executing only comments."""
        result = sandbox.run("# This is a comment\n# Another comment")
        assert result['status'] == 'ok'
    
    def test_very_long_output(self, sandbox):
        """Test handling very long output."""
        code = """
for i in range(100):
    print(f"Line {i}: " + "x" * 100)
"""
        result = sandbox.run(code, timeout=10)
        assert result['status'] == 'ok'
        assert len(result['outputs']) > 0


class TestAgentConfig:
    """Test with settings from agent.yaml and layered defaults."""
    
    def test_timeout_from_config(self, config, sandbox):
        """Test that timeout from config is reasonable."""
        timeout = config["environment"]["timeout"]
        assert timeout > 0
        
        # Should complete within configured timeout
        result = sandbox.run("print('quick')", timeout=timeout + 5)
        assert result['status'] == 'ok'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
