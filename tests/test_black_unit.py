"""
Unit tests for Black code formatter.

Tests verify Black is properly installed, configured, and accessible.
"""

import subprocess
from pathlib import Path
import pytest


def test_black_installed():
    """Test that Black 24.10.0 is installed and accessible."""
    try:
        import black
    except ImportError as e:
        pytest.fail(f"Black is not installed: {e}")

    # Verify version is 24.10.0
    version = black.__version__
    assert version == "24.10.0", f"Expected Black 24.10.0, found {version}"


def test_black_config_exists():
    """Test that Black configuration file exists."""
    config_path = Path(".config/black.toml")

    # Test fails if config file does not exist
    assert config_path.exists(), ".config/black.toml does not exist"
    assert config_path.is_file(), ".config/black.toml is not a file"


def test_black_config_valid():
    """Test that Black configuration is valid."""
    config_path = Path(".config/black.toml")

    # Test fails if config doesn't contain expected settings
    assert config_path.exists(), ".config/black.toml does not exist"

    content = config_path.read_text()

    # Check for required configuration keys
    assert "line-length = 88" in content, "line-length = 88 not found in config"
    assert "target-version = ['py311']" in content, "target-version not found in config"
    assert "include = '\\.pyi?$'" in content, "include pattern not found in config"


def test_black_check_runs():
    """Test that Black can check formatting without errors."""
    # src/ and tests/ directories should exist
    assert Path("src").exists(), "src/ directory does not exist"
    assert Path("tests").exists(), "tests/ directory does not exist"

    result = subprocess.run(
        [
            "uv",
            "run",
            "black",
            "--config",
            ".config/black.toml",
            "--check",
            "src/",
            "tests/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # We expect this to fail initially because files aren't formatted yet
    # But the black command itself should run successfully
    # Exit code 0 means files are already formatted
    # Exit code 1 means files need formatting (expected for this test)
    # Other exit codes indicate an error
    assert result.returncode in [0, 1], (
        f"Black check failed with error: {result.stderr}"
    )
