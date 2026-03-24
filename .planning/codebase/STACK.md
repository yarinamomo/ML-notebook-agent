# Technology Stack

**Analysis Date:** 2026-03-24

## Languages

**Primary:**
- Python 3.11+ - Main application and agent logic

**Secondary:**
- JavaScript - Web-based trajectory viewer (Svelte 5)

## Runtime

**Environment:**
- Python 3.11+ (minimum version specified in pyproject.toml)

**Package Manager:**
- uv - Primary Python package manager (uv.lock present)
- npm - JavaScript dependencies for viewer (`viewer/package.json`)
- Lockfiles:
  - `uv.lock` - Python dependency lockfile (present)
  - `package-lock.json` - JavaScript dependency lockfile (present)

## Frameworks

**Core:**
- mini-swe-agent 2.2.2 - Agent framework for SWE tasks, provides `DefaultAgent`, `LitellmModel`, `exceptions` modules
- litellm 1.81.6 - Unified LLM API interface, supports multiple providers (OpenAI, Anthropic, Google, custom endpoints)
- Typer 0.21.1+ - CLI framework for `main.py` command-line interface

**Testing:**
- pytest 7.0+ - Test runner and framework
- pytest-timeout 2.1+ - Test timeout handling
- pytest-cov 4.0+ - Code coverage reporting
- pytest-mock 3.10+ - Mocking support
- pytest-xdist 3.0+ - Parallel test execution

**Build/Dev:**
- Vite 6.0+ - JavaScript build tool for viewer
- @sveltejs/vite-plugin-svelte 5.0+ - Svelte integration with Vite
- Svelte 5.0+ - UI framework for trajectory viewer
- Rich 14.3+ - Terminal UI and formatting for agent execution
- Jinja2 - Template rendering for agent prompts

## Key Dependencies

**Critical:**
- docker 7.1.0 - Python SDK for Docker container management (`src/sandbox.py`)
- websocket-client 1.9.0+ - WebSocket communication with Jupyter kernels
- ipykernel 7.1.0+ - Jupyter kernel protocol client
- jupyter-kernel-client 0.8.0+ - Jupyter kernel messaging
- nbformat 5.10.4+ - Jupyter notebook format parsing and manipulation
- python-dotenv 1.0+ - Environment variable loading from `.env`
- pydantic - Data validation via BaseModel (`src/notebook_environment.py`)

**Infrastructure:**
- requests 2.32+ - HTTP client for REST API calls (Jupyter server)
- platformdirs 4.5+ - Cross-platform directory paths
- tenacity - Retry logic for Docker operations
- tiktoken - Token counting for cost tracking (via litellm)

**Viewer-specific:**
- diff 8.0.3+ - Text diffing for cell edit visualization

**Analysis Tools (Optional):**
- pandas 2.2+ - Data analysis for results
- numpy 1.24+ - Numerical computing for analysis
- matplotlib 3.8+ - Plotting for analysis
- seaborn 0.13+ - Statistical visualization

## Configuration

**Environment:**
- YAML-based configuration files in `config/` directory:
  - `defaults.yaml` - Default configuration layer
  - `agent.yaml` - Agent-specific overlay configuration
  - `baseline.yaml`, `baseline_without_cell_outputs.yaml` - Baseline mode configs
- Environment variables loaded from `.env` (gitignored)
- Layered configuration: defaults merged with run-specific overlay via `src/utils/yaml_parser.py`

**Build:**
- `pyproject.toml` - Python project configuration (PEP 621)
- `pytest.ini` - Pytest test runner configuration
- `vite.config.js` - Vite build and dev server config
- `svelte.config.js` - Svelte compiler configuration

## Platform Requirements

**Development:**
- Python 3.11+
- Docker Desktop or Docker daemon (running and accessible)
- npm (for viewer development)
- uv (recommended Python package manager)

**Production:**
- Docker host for notebook execution containers
- LLM API access (OpenAI, Anthropic, or compatible endpoint)
- File system for trajectory storage

---

*Stack analysis: 2026-03-24*