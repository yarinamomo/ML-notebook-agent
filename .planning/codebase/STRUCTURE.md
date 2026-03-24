# Codebase Structure

**Analysis Date:** 2026-03-24

## Directory Layout

```
ML-notebook-agent/
├── src/                          # Core source code
│   ├── utils/                    # Utility modules
│   │   ├── format_nb_cells.py    # Notebook cell formatting for LLM
│   │   ├── log.py                # Logging configuration
│   │   ├── nbformat_helper.py    # nbformat reading/writing
│   │   ├── nb_types.py           # Type definitions (TypedDict)
│   │   ├── notebook_command_helper.py # Command helpers
│   │   ├── retry_sandbox.py      # Retry decorators for sandbox
│   │   ├── summary_util.py       # Summary generation from messages
│   │   ├── ui.py                 # Rich UI helpers (progress bars)
│   │   └── yaml_parser.py        # Configuration loading and parsing
│   ├── benchmark.py              # Problem definition (BenchmarkProblem)
│   ├── CustomToolLitellmModel.py # LLM model with notebook tools
│   ├── notebook_environment.py   # Environment adapter for mini-swe-agent
│   ├── notebook_tools.py         # Tool definitions and action parsing
│   ├── run_agent.py              # Agent mode executor
│   ├── run_baseline.py           # Baseline mode executor
│   ├── sandbox.py                # Docker + WebSocket sandbox
│   └── ui_agent.py               # Custom agent subclass
├── config/                       # YAML configuration files
│   ├── agent.yaml                # Agent mode config (templates, prompts)
│   ├── baseline.yaml             # Baseline mode config
│   ├── defaults.yaml             # Shared default configuration
│   └── defaults_server.yaml      # Server-specific defaults
│   ├── baseline_without_cell_outputs.yaml
│   ├── baseline_without_all_outputs.yaml
│   └── without_run_code.yaml
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest configuration and fixtures
│   ├── test_compare_fixed_notebooks_unit.py
│   ├── test_notebook_environment_unit.py
│   ├── test_sandbox_integration.py # Integration tests (require Docker)
│   ├── test_sandbox_unit.py      # Unit tests (mocked)
│   ├── test_ui_agent_summary_unit.py
│   ├── test_ui_agent_unit.py
│   └── README.md                 # Test documentation
├── data_analysis/                # Analysis scripts for trajectory data
│   ├── calculate_manual_validation_results.py
│   ├── compare_fixed_notebooks.py
│   ├── edit_cell_analyze.py
│   ├── find_early_submissions.py
│   ├── run_code_analyze.py
│   ├── run_code_visualize.py
│   ├── sampler.py
│   ├── summarize_plots.py
│   └── summarize_statistics.py
├── example/                      # Test case notebooks (JunoBench)
│   └── test_case/                # Per-library test instances
│       ├── matplotlib_1/, matplotlib_2/, ...
│       ├── numpy_1/, numpy_2/, ...
│       ├── pandas_1/, pandas_2/, ...
│       ├── sklearn_1/, sklearn_2/, ...
│       ├── tensorflow_1/, ...
│       ├── torch_1/, ...
│       └── seaborn_1/, ...
├── viewer/                       # Web-based trajectory viewer
│   ├── package.json
│   ├── src/
│   │   └── lib/
│   └── (node_modules/)
├── results/                      # Run outputs (generated)
│   ├── agent/                    # Agent mode results
│   │   └── glm-4.7-355b/
│   │       ├── run_1/, run_2/, run_3/
│   │       │   ├── *.traj.json   # Trajectory message history
│   │       │   ├── *_summary.json # Execution summary
│   │       │   └── *_patched.py  # Fixed notebook as Python script
│   │       ├── analysis/         # Analysis outputs
│   │       └── plots/            # Generated plots
│   └── baseline/
├── trajectories/                 # Legacy trajectory location (mirrors results/)
├── main.py                       # CLI entry point
├── main_run_data_analysis.py     # Data analysis runner
├── preprocess_trajectories.py    # Prepare data for viewer
├── pyproject.toml                # Python project configuration
├── requirements-test.txt         # Test dependencies
├── setup_env.py                  # Environment setup helper
├── api_keys.txt                  # API keys (one per line)
├── WEBSOCKET_IMPLEMENTATION.md   # WebSocket architecture docs
└── README.md                     # Project documentation
```

## Directory Purposes

**`src/`:**
- Purpose: Core application logic for agent execution, sandbox management, and tool definitions
- Contains: Agent, environment, sandbox, runner, utilities
- Key files: `main.py`, `ui_agent.py`, `sandbox.py`, `notebook_tools.py`

**`src/utils/`:**
- Purpose: Shared utilities for formatting, logging, configuration, and type definitions
- Contains: Cell formatting for LLM, YAML parsing, summary generation, retry decorators
- Key files: `yaml_parser.py`, `format_nb_cells.py`, `summary_util.py`

**`config/`:**
- Purpose: Layered YAML configuration for agent prompts, environment settings, and model parameters
- Contains: Agent templates, system prompts, environment configs, defaults
- Key files: `defaults.yaml`, `agent.yaml`, `baseline.yaml`

**`tests/`:**
- Purpose: Comprehensive test suite with unit and integration tests
- Contains: Tests for sandbox, environment, agent, configuration
- Key files: `test_sandbox_unit.py`, `test_sandbox_integration.py`, `conftest.py`

**`data_analysis/`:**
- Purpose: Scripts to analyze trajectory data, extract patterns, and generate reports
- Contains: Analysis for run_code operations, edit_cell patterns, statistics
- Key files: `run_code_analyze.py`, `summarize_statistics.py`

**`example/test_case/`:**
- Purpose: JunoBench test cases (buggy notebooks) for evaluation
- Contains: Per-library instances (numpy, pandas, sklearn, matplotlib, etc.)
- Structure: `{library}_N/{instance_name}_reproduced.ipynb`

**`results/`:**
- Purpose: Generated outputs from agent runs (not committed to git)
- Contains: Trajectories, summaries, patched notebooks, analysis
- Structure: `{mode}/{model}/run_{N}/`

**`viewer/`:**
- Purpose: Web-based trajectory viewer for inspecting agent execution
- Contains: Svelte frontend for displaying chats and notebook diffs
- Generated: `viewer/public/data.json` from `preprocess_trajectories.py`

## Key File Locations

**Entry Points:**
- `main.py`: Primary CLI for running agents/baselines
- `preprocess_trajectories.py`: Prepares trajectory data for viewer
- `main_run_data_analysis.py`: Runs data analysis scripts

**Configuration:**
- `config/defaults.yaml`: Shared default settings (timeout, image, paths)
- `config/agent.yaml`: Agent mode prompts and templates
- `config/baseline.yaml`: Baseline mode prompts and templates

**Core Logic:**
- `src/ui_agent.py`: UI-enabled agent with timeout and logging
- `src/notebook_environment.py`: Environment adapter with tool dispatch
- `src/sandbox.py`: Docker + WebSocket kernel management
- `src/run_agent.py`: Agent mode execution orchestrator
- `src/run_baseline.py`: Baseline (single-shot) execution orchestrator

**Testing:**
- `tests/test_sandbox_integration.py`: Integration tests requiring Docker
- `tests/test_sandbox_unit.py`: Fast unit tests with mocks
- `tests/conftest.py`: Pytest fixtures and configuration

## Naming Conventions

**Files:**
- `*.py`: Python modules and scripts
- `config/*.yaml`: YAML configuration files
- `*_unit.py`: Unit test files
- `*_integration.py`: Integration test files
- `*_patched.py`: Generated fixed notebooks as Python scripts

**Directories:**
- `src/`: Source code directory
- `config/`: Configuration directory
- `tests/`: Test directory
- `results/{mode}/{model}/run_{N}/`: Per-model, per-run output directory

**Instance Names:**
- Pattern: `{library}_{number}` (e.g., `sklearn_1`, `numpy_5`)
- Examples: `torch_13`, `pandas_15`, `matplotlib_4`

## Where to Add New Code

**New Feature:**
- Primary code: `src/` (new module or extend existing)
- Tests: `tests/{feature}_unit.py`, `tests/{feature}_integration.py`

**New Component/Module:**
- Implementation: `src/{component}.py` or `src/{component}/`

**Utilities:**
- Shared helpers: `src/utils/{utility}.py`

**Configuration:**
- New configs: `config/{config_name}.yaml`

**Analysis Scripts:**
- Scripts: `data_analysis/{analysis_name}.py`

## Special Directories

**`results/`:**
- Purpose: Generated output from agent runs
- Generated: Yes
- Committed: No (in .gitignore typically)

**`trajectories/`:**
- Purpose: Legacy trajectory storage (now mirrored in results/)
- Generated: Yes
- Committed: Depends on analysis needs

**`viewer/public/data.json`:**
- Purpose: Preprocessed trajectory data for web viewer
- Generated: Yes (by `preprocess_trajectories.py`)
- Committed: Typically no for large datasets

**`example/test_case/`:**
- Purpose: Test case notebooks (JunoBench)
- Generated: No
- Committed: Yes (test fixtures)

---

*Structure analysis: 2026-03-24*