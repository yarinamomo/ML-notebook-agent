# ML Notebook Repair Agent — LLM Agents for Crash Repair

This repository contains code, data, and analysis for the paper "Beyond Crash Resolution: Evaluating LLM Agents for Repairing Machine Learning Notebooks". We build and evaluate agentic LLM systems that automatically diagnose and repair failing Jupyter notebooks, utilizing [mini-swe-agent](https://mini-swe-agent.com/) as our underlying framework. The project implements agents and baselines, reproduces experiments on the JunoBench benchmark and our self-constructed extension, and includes analysis scripts used to produce the paper results.

## Contents

- [`src/`](src/) — core agent and baseline implementations and utilities (see [`src/run_agent.py`](src/run_agent.py), [`src/run_baseline.py`](src/run_baseline.py)).
- `JunoBench/` — benchmark notebooks used in experiments. We also extended the benchmark, which can be found on [Zenodo](https://zenodo.org/records/20205838).
- [`data_analysis/`](data_analysis/) — scripts for aggregating results, computing metrics, and producing plots.
- `results_JunoBench/`, `results_new/` — Agent and baseline outputs and processed results from our experiments.
- [`config/`](config/) — YAML configs for running agents and baselines.
- [`run_all.sh`](run_all.sh) — streamlined script for running experiments.
- [`tests/`](tests/) — Unit and integration tests for the project (see [TEST_SUITE.md](TEST_SUITE.md)).

## Datasets used

- Primary benchmark: [JunoBench](https://huggingface.co/datasets/PELAB-LiU/JunoBench), excluding NBSpecific and Torch 13.
- [Supplementary JunoBench Extension](https://zenodo.org/records/20205838): We also constructed 16 new notebook instances with the most recent Kaggle notebooks from 2026-01, as well as tests for each notebook instance (JunoBench and the 16 new instances).

If you need to add other datasets, place them in a new directory and update the appropriate config (`source_path_parent`).

## Reproducing the experiments

### Prerequisites

- Python 3.9+ and a POSIX-like shell for some helper scripts (on Windows use WSL or Git Bash).
- [Docker](https://www.docker.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for Python dependency management)
- [Git LFS](https://git-lfs.com/) (required to pull the benchmark dataset)

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
# Install project dependencies
uv sync
source .venv/bin/activate
```

### 2. Pull the docker image required by notebooks in the datasets:

The implemented notebook environment for evaluating all notebooks in the datasets uses the [docker environment](https://hub.docker.com/r/yarinamomo/kaggle_python_env) from JunoBench. 

```bash
docker pull yarinamomo/kaggle_python_env:latest
```

### 3. Configure environment variables and API keys:

Create a `.env` file for local configuration using the following template:
```env
NOTEBOOK_AGENT_CONFIG_PATH=./config/agent.yaml
NOTEBOOK_AGENT_DEFAULTS_PATH=./config/defaults_local.yaml
NOTEBOOK_AGENT_LOG_LEVEL=INFO
OPENAI_API_KEY=your_key_here
```

Then, set up your keys for parallel execution:
- Edit `api_keys.txt` with your model credentials (one key per line). This file is specifically used for threading/parallel execution. Modify `run_all.sh` to ensure `API_KEYS` matches your configured text file.

### 4. Prepare the datasets:

JunoBench is included as a git submodule and uses Git LFS (Large File Storage) for the notebook data. Initialize and update it:

```bash
git submodule update --init --recursive
git lfs pull
```

To include the supplementary extension, download it from [Zenodo](https://zenodo.org/records/20205838) and extract the new instances into your local benchmark folders. You will need to explicitly point the agent to these folders in your configuration files by changing the `source_path_parent` key in your config file (e.g., `source_path_parent: JunoBench/benchmark/`, see [config/CONFIG.md](config/CONFIG.md)).

### 5. Running experiments

- To run the full benchmark pipeline across all configurations:

```bash
./run_all.sh
```

- To run a single experiment or setting:

```bash
python main.py --config config/agent.yaml
```

**Configuration Mechanics**

The runner ([`main.py`](main.py)) relies on a layered configuration system that merges a defaults file with a run-specific configuration. *For full details on all available configurations and extensions, see [config/CONFIG.md](config/CONFIG.md).*

Notes
- Long-running jobs will create per-run outputs under the configured folder in the config files, e.g. `trajectories/[setting_name]/[model]/[run_id]`.


## Results & analysis

Metrics reported in the paper can be reproduced using ([`main_run_data_analysis.py`](main_run_data_analysis.py)).

## Acknowledgments

This project is built on top of the [mini-swe-agent](https://mini-swe-agent.com/) framework. We thank the authors and maintainers for providing a robust foundation for agentic software engineering.

<!-- ## Releasing and citing

If you use this code or dataset in your research, please cite the associated paper and include a link to this repository. Contact the authors via the repository maintainer listed in the paper for further details.

## Contact

For questions about reproducing results or running the code, open an issue or contact the maintainers listed in the paper. -->
