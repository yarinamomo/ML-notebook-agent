# NotebookSandbox

A Docker-based Jupyter notebook execution sandbox, or custom environment adapter (`NotebookEnvironment`) with [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) integration for AI-powered notebook debugging.

## Quick Start

1. `python setup_env.py` (to setup your OpenAI API key)
2. Run examples: `python main.py`

## Installation

```bash
# with pip
pip install -e .
```

## Usage

- **[Example Notebook](nb_sandbox_example.ipynb)** - Basic sandbox usage
- **[Examples Script](mini_swe_agent_examples.py)** - Mini-swe-agent integration examples

## Requirements

- Docker Desktop (running)
- Python 3.11+
- Docker image with Jupyter (e.g., `yarinamomo/kaggle_python_env`)
- OpenAI/Anthropic/Google API key (for mini-swe-agent features)

## Code Quality

This project uses Black code formatter for consistent code style.

### Installation

```bash
# Install development dependencies (including Black)
uv pip install -e ".[dev]"
```

### Manual Formatting

```bash
# Format all Python files
black --config .config/black.toml src/ tests/

# Check formatting without applying changes
black --config .config/black.toml --check src/ tests/
```

### Pre-commit Hooks

Black is configured as a pre-commit hook to automatically format files before commits.

```bash
# Install pre-commit hooks (runs once)
pre-commit install

# Run pre-commit hooks manually
pre-commit run black --all-files
```

### Configuration

Black configuration is in `.config/black.toml`:
- Line length: 88 characters
- Target Python version: 3.11
- Matches project conventions in codebase

## Trajectory Viewer

A web-based tool for inspecting agent trajectories. Shows the chat history on the left and the notebook state on the right, with inline git-style diffs for cell edits.

### Setup

```bash
# 1. Preprocess trajectory data into a single JSON file
python3 preprocess_trajectories.py

# 2. Install viewer dependencies (once)
cd viewer && npm install

# 3. Start the dev server
npm run dev
```

The viewer opens at [http://localhost:5173](http://localhost:5173).

### Features

- **Selector bar** — Pick model, library, run, and instance from cascading dropdowns. Metadata (status, cost, time, edits) is shown inline.
- **Chat panel (left)** — All agent steps displayed as scrollable cards with reasoning, action, and observation. Click or use arrow keys to step through; the active step is highlighted and auto-scrolled.
- **Notebook panel (right)** — All notebook cells shown at a glance. When the active step is a cell edit, the viewer auto-scrolls to that cell and displays an inline line-by-line diff.
- **Keyboard navigation** — Arrow up/down to move between steps.

### Regenerating Data

After new agent runs are added to `trajectories/`, re-run the preprocessor:

```bash
python3 preprocess_trajectories.py
```

This reads all `*.traj.json`, `*_summary.json`, and `*_patched.py` files and writes `viewer/public/data.json`.
