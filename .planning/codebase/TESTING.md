# Testing Patterns

**Analysis Date:** 2026-03-24

## Test Framework

**Runner:**
- **pytest** (7.0.0+)
- Config: `pytest.ini` and `conftest.py` in `tests/`

**Assertion Library:**
- pytest built-in assertions (no separate library)

**Run Commands:**
```bash
pytest                            # Run all tests
pytest -v                         # Verbose output
pytest -m unit                    # Run only unit tests
pytest -m integration             # Run only integration tests
pytest -m "not slow"              # Run all non-slow tests
pytest -x                         # Stop on first failure
pytest --cov=src                  # Run with coverage
pytest --cov-report=html          # Generate HTML coverage report
```

## Test File Organization

**Location:**
- Separated from source code in `tests/` directory
- No test files colocated with source (standard pytest pattern)

**Naming:**
- Pattern: `test_<module>_<type>.py`
- Examples:
  - `test_sandbox_unit.py` - Unit tests for sandbox
  - `test_sandbox_integration.py` - Integration tests for sandbox
  - `test_ui_agent_unit.py` - Unit tests for UI agent
  - `test_notebook_environment_unit.py` - Unit tests for notebook environment
  - `test_ui_agent_summary_unit.py` - Unit tests for summary generation

**Structure:**
```
tests/
├── conftest.py                   # Shared fixtures and configuration
├── test_sandbox_unit.py          # Unit tests for DockerSandbox
├── test_sandbox_integration.py   # Integration tests requiring Docker
├── test_ui_agent_unit.py         # Unit tests for UiAgent
├── test_notebook_environment_unit.py  # Unit tests for NotebookEnvironment
└── test_ui_agent_summary_unit.py     # Unit tests for summary generation
```

## Test Structure

**Suite Organization:**
```python
"""
Unit tests for DockerSandbox class.

Tests core functionality with mocked Docker and WebSocket components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.sandbox import DockerSandbox

class TestDockerSandboxInit:
    """Test DockerSandbox initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        with patch('src.sandbox.docker.from_env'):
            sandbox = DockerSandbox("test-image")
            assert sandbox.image_name == "test-image"
            assert sandbox.port == 8888
```

**Class-based Organization:**
- Group related tests in test classes: `class Test<Feature>:`
- One class per major feature area
- Descriptive method names: `def test_<scenario>():`

**Patterns:**
- **Setup:** Create test fixtures in `conftest.py` or as pytest fixtures within test files
- **Teardown:** Use `pytest.fixture` yields for cleanup, or `with` statements for resource management
- **Assertion:** Use plain pytest assertions (not unittest assertions)
- **Isolation:** Each test is independent; use mocks to avoid external dependencies

**Fixture Pattern:**
```python
@pytest.fixture
def env(monkeypatch, tmp_path):
    """Create a NotebookEnvironment for testing."""
    monkeypatch.setattr("src.notebook_environment.BenchmarkProblem", FakeBenchmarkProblem)
    return NotebookEnvironment(instance_name="dummy", output_dir=tmp_path)
```

## Mocking

**Framework:**
- `unittest.mock` (built-in Python library)
- Use `@patch` decorator for function/class mocking
- Use `Mock` and `MagicMock` for test doubles

**Patterns:**
```python
from unittest.mock import Mock, patch

# Patch external dependencies
@patch('src.sandbox.docker.from_env')
def test_init_default_values(self, mock_docker):
    sandbox = DockerSandbox("test-image")
    assert sandbox.image_name == "test-image"

# Patch multiple items
@patch('src.sandbox.DockerSandbox.restart_kernel')
@patch('src.sandbox.DockerSandbox._start_kernel')
def test_restart_kernel_cleans_up(self, mock_start_kernel, mock_restart):
    # Test implementation

# Create mock objects
mock_response = Mock()
mock_response.json.return_value = {'id': 'kernel-123'}

# Mock configuration
mock_ws = Mock()
mock_ws.sock = Mock()
mock_ws.sock.connected = True
```

**What to Mock:**
- External service calls (Docker, Jupyter kernel, WebSocket)
- Filesystem operations when testing logic, not I/O
- Network requests (HTTP, WebSocket connections)
- Heavy dependencies that don't need to be tested directly
- Third-party library interactions

**What NOT to Mock:**
- Business logic you're actually testing
- Simple data structures (lists, dicts)
- Functions/classes under test themselves
- Data validation logic (test it against real validators)
- Configuration loading (test with actual config files)

## Fixtures and Factories

**Test Data:**
```python
class StubModel:
    """Simple stub for model interface."""
    def __init__(self, responses=None):
        self._responses = list(responses or [])

    def query(self, messages, **kwargs):
        if not self._responses:
            return {
                "role": "exit",
                "content": "done",
                "extra": {"exit_status": "Submitted", "submission": ""},
            }
        return self._responses.pop(0)


class FakeBenchmarkProblem:
    """Minimal benchmark stub to control notebook execution outcomes in tests."""

    def __init__(self, *args, **kwargs):
        self._cell_count = 1

    def get_cell_count(self):
        return self._cell_count
```

**Location:**
- Test stubs: Defined within test files using the class
- Reusable fixtures: In `tests/conftest.py`

**Shared Fixtures (conftest.py):**
```python
@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    return {
        "integration_tests": True,
        "unit_tests": True,
        "timeout": 30,
    }

@pytest.fixture(autouse=True)
def reset_execution_state():
    """Reset execution state between tests."""
    yield
    # Cleanup after each test

@pytest.fixture
def config():
    """Load configuration from agent.yaml with layered defaults."""
    config_path = Path(__file__).parent.parent / "config" / "agent.yaml"
    return load_config(config_path, DEFAULTS_CONFIG)
```

## Coverage

**Requirements:**
- Configured via `pytest-cov` (4.0.0+)
- No explicit coverage percentage requirements documented

**Coverage Configuration:**
```ini
[coverage:run]
source = src
omit =
    */tests/*
    */__pycache__/*
    */site-packages/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod
```

**View Coverage:**
```bash
pytest --cov=src --cov-report=term-missing
pytest --cov=src --cov-report=html
# Open htmlcov/index.html for detailed report
```

## Test Types

**Unit Tests:**
- Scope: Test individual functions, classes, and modules in isolation
- Marked with: `@pytest.mark.unit` (auto-applied from filename)
- No external dependencies: All external calls are mocked
- Fast execution: Should run in milliseconds
- Examples:
  - `test_sandbox_unit.py` - Tests DockerSandbox without actual Docker
  - `test_ui_agent_unit.py` - Tests agent logic with stubbed model/env
  - `test_notebook_environment_unit.py` - Tests environment dispatch logic

**Integration Tests:**
- Scope: Test interaction between components and external systems
- Marked with: `@pytest.mark.integration` (auto-applied from filename)
- Require: Docker daemon, network connectivity
- Slower execution: May take seconds to complete
- Examples:
  - `test_sandbox_integration.py` - Tests with real Docker and Jupyter

**E2E Tests:**
- Not explicitly used
- Integration tests serve this purpose
- No separate E2E framework configured

## Test Markers

**Configured Markers:**
```python
@pytest.mark.unit          # Unit tests (fast, no Docker)
@pytest.mark.integration   # Integration tests (requires Docker)
@pytest.mark.slow          # Slow running tests
@pytest.mark.docker        # Tests requiring Docker
```

**Marker Application:**
- Auto-applied by path in `pytest_collection_modifyitems` (conftest.py)
- Applied by name for slow tests:
```python
slow_test_names = [
    "test_timeout",
    "test_disconnect",
    "sequential",
    "multiple",
]
if any(name in item.name for name in slow_test_names):
    item.add_marker(pytest.mark.slow)
```

## Common Patterns

**Async Testing:**
- Not applicable (codebase uses threading, not async/await)

**Error Testing:**
```python
def test_interrupt_kernel_no_kernel_id(self):
    """Test interrupt fails when no kernel_id."""
    with patch('src.sandbox.docker.from_env'):
        sandbox = DockerSandbox("test-image")
        sandbox.kernel_id = None

        with pytest.raises(RuntimeError, match="No kernel ID"):
            sandbox._interrupt_kernel()
```

**Timeout Testing:**
```python
def test_execute_with_timeout(self, sandbox):
    """Test code execution with timeout."""
    code = "import time\ntime.sleep(5)\n"
    result = sandbox.run(code, timeout=1)
    assert result['status'] == 'timeout'
```

**Exception Testing:**
```python
def test_run_raises_agent_timeout_before_model_call(quiet_ui):
    """Test that AgentTimeout is raised."""
    agent = UiAgent(model=model, env=env, total_timeout=1)
    agent._start_time = time.monotonic() - 5

    with pytest.raises(AgentTimeout) as exc_info:
        agent.query()

    timeout_message = exc_info.value.messages[0]
    assert timeout_message["content"] == "AgentTimeout"
```

**Using Monkeypatch:**
```python
@pytest.fixture
def quiet_ui(monkeypatch):
    """Suppress UI output for cleaner test logs."""
    nullcontext = contextlib.nullcontext
    monkeypatch.setattr("src.ui_agent.ui.wait_llm", lambda: nullcontext())
    monkeypatch.setattr("src.ui_agent.ui.wait_tool", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr("src.ui_agent.ui.system", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui_agent.ui.agent", lambda *_args, **_kwargs: None)
```

**Testing with Temporary Paths:**
```python
def test_save_summary_includes_operations(quiet_ui, tmp_path):
    """Test summary file creation with actual file write."""
    output_path = tmp_path / "summary.json"
    summary = agent.save_summary(output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == summary
```

## Test Configuration

**Test Discovery:**
```ini
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
testpaths = tests
```

**Timeout:**
- Default timeout: 300 seconds (5 minutes)
- Configured in `pytest.ini` via `pytest-timeout` plugin
- Timeout applies per test, not for entire suite

**Logging in Tests:**
```ini
log_cli = false
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

## Test Dependencies

**Test Dependencies:**
```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-timeout>=2.1.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-xdist>=3.0.0",
]
```

**Key Libraries:**
- `pytest-timeout`: Adds test timeout support
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Enhanced mocking utilities
- `pytest-xdist`: Parallel test execution (not explicitly used in commands)

## Running Tests by Category

**Unit Tests Only:**
```bash
pytest -m unit -v
```

**Integration Tests Only:**
```bash
pytest -m integration -v
```

**Fast Tests (exclude slow):**
```bash
pytest -m "not slow"
```

**Skip Integration:**
```bash
pytest -m "not integration"
```

---

*Testing analysis: 2026-03-24*