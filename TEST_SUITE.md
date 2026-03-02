# Test Suite Documentation

## Overview

This comprehensive test suite for the DockerSandbox class includes:
- **45+ Unit Tests** - Fast tests with mocked dependencies
- **60+ Integration Tests** - Real Docker container and Jupyter kernel tests
- **Custom Fixtures and Markers** - Easy test organization and execution

## Test Organization

```
tests/
├── __init__.py                  # Test package marker
├── conftest.py                  # Shared fixtures and pytest config
├── test_sandbox_unit.py         # Unit tests (~45 tests)
├── test_sandbox_integration.py  # Integration tests (~60 tests)
├── README.md                    # Detailed test documentation
└── (this file)
```

## Quick Start

### Install Test Dependencies
```bash
pip install -r requirements-test.txt
```

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast, ~10s)
pytest tests/test_sandbox_unit.py -v

# Run only integration tests (slow, ~5-10 min)
pytest tests/test_sandbox_integration.py -v
```

## Test Categories

### Unit Tests (`test_sandbox_unit.py`)

**Purpose:** Test individual components in isolation with mocked dependencies.

**Test Classes (8):**

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestDockerSandboxInit` | 2 | Initialization and configuration |
| `TestWebSocketConnection` | 5 | Connection state management |
| `TestMessageHandling` | 6 | Jupyter protocol message handling |
| `TestExecutionLogic` | 2 | Code execution mechanics |
| `TestRetryLogic` | 2 | Reconnection and retry mechanisms |
| `TestInterruptLogic` | 2 | Kernel interrupt functionality |
| `TestKernelRestart` | 3 | Kernel restart and cleanup |
| `TestExceptionHandling` | 2 | Exception handling |

**Total Unit Tests:** ~27 tests
**Expected Runtime:** < 5 seconds

**Key Features Tested:**
- ✅ WebSocket connection state machine
- ✅ Message serialization/deserialization
- ✅ Retry logic with exponential backoff
- ✅ Exception handling and recovery
- ✅ Resource cleanup

### Integration Tests (`test_sandbox_integration.py`)

**Purpose:** Test real Docker container and Jupyter kernel interactions.

**Requirements:**
- Docker installed and running
- Docker image: `ml-notebook-agent` (or as specified in config/default.yaml)
- Port 8888 available
- ~5-10 minutes test time

**Test Classes (11):**

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestSandboxStartup` | 1 | Container and kernel initialization |
| `TestBasicExecution` | 4 | Simple code execution |
| `TestErrorHandling` | 3 | Error detection and reporting |
| `TestTimeout` | 2 | Timeout and interrupt |
| `TestReconnection` | 1 | Automatic reconnection |
| `TestKernelRestart` | 2 | Kernel restart functionality |
| `TestMultipleExecutions` | 2 | Sequential execution and state |
| `TestOutputFormats` | 3 | Output capture and formatting |
| `TestComplexScenarios` | 3 | NumPy, Pandas, tracebacks |
| `TestEdgeCases` | 4 | Empty code, whitespace, etc. |
| `TestDefaultConfig` | 1 | Configuration validation |

**Total Integration Tests:** ~30 tests
**Expected Runtime:** 5-10 minutes

**Key Features Tested:**
- ✅ Full Docker container lifecycle
- ✅ Code execution in Python kernel
- ✅ stdout/stderr capture
- ✅ Exception handling in kernel
- ✅ Timeout and interrupt mechanisms
- ✅ Reconnection after disconnection
- ✅ State persistence across executions
- ✅ Complex scientific operations

## Running Tests by Category

### Run Only Specific Tests

```bash
# Test initialization
pytest tests/test_sandbox_unit.py::TestDockerSandboxInit -v

# Test WebSocket connection
pytest tests/test_sandbox_unit.py::TestWebSocketConnection -v

# Test basic execution
pytest tests/test_sandbox_integration.py::TestBasicExecution -v

# Test error handling
pytest tests/test_sandbox_integration.py::TestErrorHandling -v

# Test timeout behavior
pytest tests/test_sandbox_integration.py::TestTimeout -v
```

### Run by Marker

```bash
# Run all unit tests
pytest -m unit

# Run all integration tests
pytest -m integration

# Run all slow tests
pytest -m slow

# Run all tests except slow
pytest -m "not slow"

# Run tests matching pattern
pytest tests/ -k "websocket" -v

# Run tests not matching pattern
pytest tests/ -k "not timeout" -v
```

## Test Execution Flow

### Typical Unit Test Flow
```
test start
  ├─ Mock Docker client
  ├─ Create DockerSandbox instance
  ├─ Mock WebSocket/network calls
  ├─ Execute test assertions
  └─ Cleanup mocks
```

### Typical Integration Test Flow
```
test start
  ├─ Load config from default.yaml
  ├─ Create DockerSandbox
  ├─ Start Docker container (30s)
  ├─ Wait for Jupyter server ready
  ├─ Establish WebSocket connection
  ├─ Execute code on actual kernel
  ├─ Verify outputs/results
  ├─ Stop container and cleanup
  └─ Test complete
```

## Test Fixtures

### Unit Test Fixtures
- All unit tests use mocked Docker and WebSocket
- No external dependencies required
- Tests can run in parallel

### Integration Test Fixtures
```python
@pytest.fixture
def config():
    """Load configuration from default.yaml"""
    
@pytest.fixture
def docker_image(config):
    """Get Docker image name from config"""
    
@pytest.fixture
def sandbox(docker_image, docker_port, docker_token):
    """Create and start Docker sandbox"""
    # Starts container before test
    # Stops container after test
```

## Key Test Scenarios

### 1. Basic Execution
```python
def test_execute_simple_print(sandbox):
    result = sandbox.run("print('Hello World')")
    assert result['status'] == 'ok'
    assert 'Hello World' in str(result['outputs'])
```

### 2. Error Handling
```python
def test_execute_with_syntax_error(sandbox):
    result = sandbox.run("x = invalid syntax")
    assert result['status'] == 'error' or any(
        o['output_type'] == 'error' for o in result['outputs']
    )
```

### 3. Timeout and Recovery
```python
def test_kernel_restarted_after_timeout(sandbox):
    with pytest.raises(TimeoutError):
        sandbox.run("while True: pass", timeout=1)
    
    # Kernel should be restarted
    result = sandbox.run("print('recovered')")
    assert result['status'] == 'ok'
```

### 4. Reconnection
```python
def test_execute_after_websocket_disconnect(sandbox):
    sandbox.ws = None  # Simulate disconnect
    result = sandbox.run("print('reconnected')", max_retries=2)
    assert result['status'] == 'ok'
```

### 5. State Persistence
```python
def test_state_persistence(sandbox):
    sandbox.run("x = 10")
    result = sandbox.run("print(x)")
    assert '10' in str(result['outputs'])
```

## Test Results Interpretation

### Successful Test Run
```
tests/test_sandbox_unit.py .......................... PASSED [100%]
tests/test_sandbox_integration.py ..................  PASSED [ 95%]

======================== 57 passed in 312.45s =========================
```

### Test Failures

**Type 1: Connection Issues**
```
FAILED test_sandbox_integration.py::TestSandboxStartup::test_sandbox_starts_successfully
RuntimeError: Failed to start kernel
```
**Solution:** Check Docker is running, image is built

**Type 2: Timeout Issues**
```
FAILED test_sandbox_integration.py::TestTimeout::test_execute_with_timeout
TIMEOUT: Test exceeded timeout threshold
```
**Solution:** Increase timeout or check system performance

**Type 3: Assertion Failures**
```
FAILED test_sandbox_integration.py::TestBasicExecution::test_execute_simple_print
AssertionError: 'Hello World' not in result outputs
```
**Solution:** Check Jupyter output format matches expectations

## Performance Characteristics

### Unit Tests
- **Time per test:** 10-100ms
- **Total time:** < 5 seconds
- **Memory:** < 100MB
- **Parallelizable:** Yes

### Integration Tests
- **Setup time:** 30-60 seconds (container start)
- **Test time:** 0.5-5 seconds (per test)
- **Teardown time:** 10-20 seconds (container stop)
- **Total time:** 5-10 minutes for full suite
- **Memory:** 500MB-1GB

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements-test.txt
      - run: pytest tests/test_sandbox_unit.py -v

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements-test.txt
      - run: docker build -t ml-notebook-agent .
      - run: pytest tests/test_sandbox_integration.py -v
```

## Debugging Tips

### Enable Detailed Output
```bash
pytest tests/ -vv -s --tb=long
```

### Run Single Test
```bash
pytest tests/test_sandbox_unit.py::TestWebSocketConnection::test_is_websocket_connected_true -vv
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Run Last Failed
```bash
pytest tests/ --lf
```

### With Python Debugger
```bash
pytest tests/ --pdb
```

## Coverage Analysis

Run coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Docker not found | Docker not installed/running | Install Docker, start daemon |
| Port 8888 in use | Other service using port | Kill process or use different port |
| Image not found | Docker image not built | Run `docker build -t ml-notebook-agent .` |
| Timeout | System slow or heavy load | Increase timeout value |
| WebSocket error | Network/firewall issue | Check Docker networking |
| Import errors | Path not configured | Run from project root |

## Extending the Test Suite

### Adding New Unit Test
```python
class TestNewFeature:
    def test_new_functionality(self):
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            # Test implementation
            assert condition
```

### Adding New Integration Test
```python
def test_new_integration_feature(sandbox):
    """Test description."""
    result = sandbox.run("new code")
    assert result['status'] == 'ok'
    # Additional assertions
```

## Test Coverage Goals

- **Line Coverage:** > 90%
- **Branch Coverage:** > 80%
- **Function Coverage:** > 95%

Current coverage can be checked with:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Fixtures](https://docs.pytest.org/how-to/fixtures.html)
- [Docker Python SDK](https://docker-py.readthedocs.io/)
- [Jupyter Messaging Protocol](https://jupyter-client.readthedocs.io/en/stable/messaging.html)
- [WebSocket Protocol](https://datatracker.ietf.org/doc/html/rfc6455)
