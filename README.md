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
