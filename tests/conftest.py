"""
Pytest configuration and shared fixtures for sandbox tests.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test requiring Docker"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test module."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        slow_test_names = [
            "test_timeout",
            "test_disconnect",
            "sequential",
            "multiple",
        ]
        if any(name in item.name for name in slow_test_names):
            item.add_marker(pytest.mark.slow)
