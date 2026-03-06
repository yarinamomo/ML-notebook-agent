# Test Suite for ML-Notebook-Agent

This directory contains a comprehensive test suite for the DockerSandbox class, including both unit tests and integration tests.

## Test Structure

### Unit Tests (`test_sandbox_unit.py`)
Tests for individual components with mocked dependencies. These tests are fast and don't require Docker.

**Test Classes:**
- `TestDockerSandboxInit` - Initialization and default values
- `TestWebSocketConnection` - WebSocket connection state checking
- `TestMessageHandling` - Jupyter message type handling (stream, error, result, etc.)
- `TestExecutionLogic` - Code execution and message flow
- `TestRetryLogic` - Reconnection and retry mechanisms
- `TestInterruptLogic` - Kernel interruption via REST API
- `TestKernelRestart` - Kernel restart functionality
- `TestExceptionHandling` - Exception handling and recovery

### Integration Tests (`test_sandbox_integration.py`)
Tests for real Docker container and Jupyter kernel interactions. Requires the Docker image specified by `config/agent.yaml` together with layered defaults.

**Test Classes:**
- `TestSandboxStartup` - Container and kernel initialization
- `TestBasicExecution` - Simple code execution (print, variables, imports)
- `TestErrorHandling` - Syntax/runtime error handling
- `TestTimeout` - Execution timeout and interruption
- `TestReconnection` - WebSocket disconnection recovery
- `TestKernelRestart` - Kernel restart and fresh state
- `TestMultipleExecutions` - Sequential executions and state persistence
- `TestOutputFormats` - stdout/stderr capture and multiple outputs
- `TestComplexScenarios` - NumPy, Pandas, and real-world operations
- `TestEdgeCases` - Empty code, whitespace, comments, long output
- `TestAgentConfig` - Configuration from agent.yaml and layered defaults

## Installation

Install test dependencies:

```bash
pip install pytest pytest-timeout pyyaml
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Only Unit Tests (Fast)
```bash
pytest tests/test_sandbox_unit.py -v
```

### Run Only Integration Tests (Requires Docker)
```bash
pytest tests/test_sandbox_integration.py -v
```

### Run Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Class
```bash
pytest tests/test_sandbox_unit.py::TestWebSocketConnection -v
```

### Run Specific Test Function
```bash
pytest tests/test_sandbox_unit.py::TestWebSocketConnection::test_is_websocket_connected_true -v
```

### Run with Output Display
```bash
pytest tests/ -v -s
```

Options:
- `-v` - Verbose output
- `-s` - Show print statements and output
- `--tb=short` - Shorter traceback format
- `--tb=long` - Full traceback format
- `-x` - Stop on first failure
- `--lf` - Run last failed tests
- `--ff` - Run failed tests first

## Integration Test Prerequisites

### Docker Image
The integration tests require the Docker image specified by `config/agent.yaml` together with layered defaults:
```yaml
environment:
  docker_image_name: ml-notebook-agent
```

**Build the image (if needed):**
```bash
docker build -f Dockerfile -t ml-notebook-agent .
```

### Config File
Integration tests load configuration from `config/agent.yaml` via layered defaults. Key settings:
- `docker_image_name` - The Docker image to use
- `timeout` - Default timeout for executions
- `docker_start_command` - Jupyter server startup command

## Test Coverage

### Unit Test Coverage
- ✅ Initialization and configuration
- ✅ WebSocket connection state management
- ✅ Jupyter message protocol handling
- ✅ Execution message creation and processing
- ✅ Reconnection and retry logic with exponential backoff
- ✅ Exception handling and error recovery
- ✅ Kernel restart operations
- ✅ REST API calls for interrupt and kernel management

### Integration Test Coverage
- ✅ Full Docker container startup and shutdown
- ✅ Jupyter server initialization
- ✅ WebSocket connection establishment
- ✅ Basic code execution (print, variables, imports)
- ✅ Error handling (syntax, runtime, undefined variables)
- ✅ Timeout and interrupt mechanisms
- ✅ Automatic reconnection after disconnection
- ✅ Kernel restart and fresh state
- ✅ Sequential executions with state persistence
- ✅ Output capture (stdout, stderr, multiple outputs)
- ✅ Complex scientific computing operations
- ✅ Edge cases (empty code, whitespace, comments)
- ✅ Very long output handling

## Key Test Scenarios

### Timeout Handling
```python
def test_execute_with_timeout(sandbox):
    with pytest.raises(TimeoutError):
        sandbox.run("while True: pass", timeout=1)
```

### Reconnection
```python
def test_execute_after_websocket_disconnect(sandbox):
    sandbox.ws = None  # Simulate disconnect
    result = sandbox.run("print('reconnected')", max_retries=2)
    assert result['status'] == 'ok'
```

### Error Recovery
```python
def test_kernel_restarted_after_timeout(sandbox):
    with pytest.raises(TimeoutError):
        sandbox.run("while True: pass", timeout=1)
    # Kernel should be restarted
    result = sandbox.run("print('ok')")
    assert result['status'] == 'ok'
```

### State Persistence
```python
def test_state_persistence_across_executions(sandbox):
    sandbox.run("counter = 0")
    sandbox.run("counter += 1")
    result = sandbox.run("print(counter)")
    assert "1" in str(result['outputs'])
```

## Debugging Tests

### Run with Fine-Grained Logging
```bash
pytest tests/ -v -s --log-cli-level=DEBUG
```

### Run with Pdb on Failure
```bash
pytest tests/ --pdb
```

### Run with Pdb on Error
```bash
pytest tests/ --pdbcls=IPython.terminal.debugger:TerminalPdb
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/test_sandbox_unit.py -v
```

## Notes

- **Unit tests** should complete in seconds
- **Integration tests** may take several minutes (container startup, execution, shutdown)
- Integration tests require Docker to be installed and running
- Tests clean up Docker resources automatically (containers, kernels)
- Failed integration tests may leave dangling containers (clean up with `docker ps -a`)

## Troubleshooting

### Docker Connection Issues
```bash
# Check Docker is running
docker ps

# List containers
docker ps -a

# Remove dangling containers
docker container prune
```

### Timeout Issues
If integration tests timeout, increase timeout in test or adjust system settings:
```python
sandbox.run(code, timeout=60)  # Increase timeout
```

### WebSocket Connection Issues
Check that Docker ports are not bound:
```bash
lsof -i :8888  # Check what's using port 8888
```

### Import Issues
Ensure project root is in Python path:
```bash
cd /path/to/ml-notebook-agent
pytest tests/
```

## Contributing

When adding new functionality to `DockerSandbox`:
1. Add corresponding unit tests to `test_sandbox_unit.py`
2. Add integration tests to `test_sandbox_integration.py`
3. Ensure all tests pass before submitting PR
4. Update this README with new test scenarios
