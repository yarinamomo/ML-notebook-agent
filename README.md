# ML Notebook Repair Agent — LLM Agents for Crash Repair

This repository contains code, data, and analysis for the paper "Beyond Crash Resolution: Evaluating LLM Agents for Repairing Machine Learning Notebooks". We build and evaluate agentic LLM systems that automatically diagnose and repair failing Jupyter notebooks. The project implements agents and baselines, reproduces experiments on the JunoBench benchmark and our self-constructed extension, and includes analysis scripts used to produce the paper results.

## Contents

- `src/` — core agent and baseline implementations and utilities (see `src/run_agent.py`, `src/run_baseline.py`).
- `JunoBench/` — benchmark notebooks used in experiments. We also extended the benchmark, which can be found on [Zenodo](https://zenodo.org/records/20205838).
- `data_analysis/` — scripts for aggregating results, computing metrics, and producing plots.
- `results_JunoBench/`, `results_new/` — Agent and baseline outputs and processed results from our experiments.
- `config/` — YAML configs for running agents and baselines.
- `run_all.sh` — streamlined script for running experiments.

## Datasets used

- Primary benchmark: [JunoBench](https://huggingface.co/datasets/PELAB-LiU/JunoBench), excluding NBSpecific and Torch 13.
- [Supplementary JunoBench Extension](https://zenodo.org/records/20205838): We also constructed 16 new notebook instances with the most recent Kaggle notebooks from 2026-01, as well as tests for each notebook instance (JunoBench and the 16 new instances).

If you need to add other datasets, place them in a new directory and update the appropriate config (`source_path_parent`).

## Reproducing the experiments

Prerequisites

- Python 3.9+ and a POSIX-like shell for some helper scripts (on Windows use WSL or Git Bash).
- API keys: place OpenAI/other locally hosted model keys in `api_keys.txt` on local machine as needed.
- `.env`: environment file for local configuration (e.g., NOTEBOOK_AGENT_DEFAULTS_PATH = ./config/defaults_local.yaml).
- Local notebook environment: The implemented notebook environment for evaluating all notebooks in the datsets uses the [docker environment](https://hub.docker.com/repository/docker/yarinamomo/kaggle_python_env/tags/latest/sha256-73380761b1f37a83aef2c247a9d725c796c6196abf14bccc92b92b25c7eb81b9) from JunoBench (sha256:73380761b1f37a83aef2c247a9d725c796c6196abf14bccc92b92b25c7eb81b9)

Quick setup

1. Create and activate a virtual environment, then install core requirements:

```bash
# On Linux
uv sync
source .venv/bin/activate
```

2. Pull the docker image required by notebooks in the datasets:

```bash
docker pull yarinamomo/kaggle_python_env:latest
```

3. Populate API keys:

 - Edit `api_keys.txt` with your model credentials (one key per line as expected by the code) and modify `run_all.sh` (API_KEYS) with the corresponding name.

4. Prepare the dataset (clone JunoBench and download the extension; place the extension in corresponding benchmark folders)


Running experiments

- To run the full benchmark pipeline (agents/baselines + evaluation):

```bash
./run_all.sh
```

- To run only the test-based evaluation for a specific setting (e.g., agent):

```bash
python data_analysis/run_all_patched_notebooks.py -c config/agent.yaml
```

Notes

- Use `--config` to select different settings (see `config/` for available settings).
- Long-running jobs will create per-run outputs under configured folder in the config files, e.g. `trajectories/[setting_name]/[model_name]/[run_id]`.


## Results & analysis

Metrics reported in the paper can be reproduced using (`main_run_data_analysis.py`).

<!-- ## Releasing and citing

If you use this code or dataset in your research, please cite the associated paper and include a link to this repository. Contact the authors via the repository maintainer listed in the paper for further details.

## Contact

For questions about reproducing results or running the code, open an issue or contact the maintainers listed in the paper. -->
