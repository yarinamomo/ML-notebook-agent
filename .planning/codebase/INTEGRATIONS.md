# External Integrations

**Analysis Date:** 2026-03-24

## APIs & External Services

**LLM Providers:**
- OpenAI API - Primary LLM provider for agent reasoning
  - SDK/Client: litellm统一接口
  - Auth: `OPENAI_API_KEY` or `api_key` in .env/environment
  - Configuration: Customizable via `model_kwargs.api_base` for custom endpoints
  - Used in: `src/CustomToolLitellmModel.py`

- Anthropic API - Alternative LLM provider
  - SDK/Client: litellm统一接口
  - Auth: `ANTHROPIC_API_KEY` in .env/environment

- Custom/Third-party LLM endpoints - Supported via litellm
  - Example: `https://aqueduct.ai.datalab.tuwien.ac.at/v1` (TU Wien)
  - Configured via `model_kwargs.custom_llm_provider` and `model_kwargs.api_base`
  - Custom cost tracking: `input_cost_per_token`, `output_cost_per_token`

**Jupyter Server API:**
- Local Docker-based Jupyter server
  - REST API endpoints:
    - `GET /api/status` - Health check
    - `POST /api/kernels` - Create new kernel
    - `DELETE /api/kernels/{id}` - Delete kernel
    - `POST /api/kernels/{id}/interrupt` - Interrupt kernel execution
  - Auth: Token-based (default: `Super_Duper_Secret_Token`)
  - Connection: HTTP on port 8888 (configurable)
  - Used in: `src/sandbox.py`

## Data Storage

**Databases:**
- None - No traditional database used

**File Storage:**
- Local filesystem - All data stored locally
  - Trajectories: `./trajectories/{mode}/{model_name}/run_{n}/`
  - Results: `./results/{mode}/...`
  - Configuration: `./config/` directory

**Notebook Storage:**
- Local IPython notebook files (.ipynb)
  - parsed and manipulated via nbformat library
  - Path: `example/JunoBench/` for benchmark instances

**Caching:**
- None - No caching layer implemented

## Authentication & Identity

**Auth Provider:**
- Custom - API key authentication for LLM services
  - Implementation: Environment variables or file-based API keys (`api_keys.txt`)
  - Multi-threading: Each thread uses separate API key from file
  - Keys stored in `.env` (gitignored) or loaded from `api_keys.txt`

**Token Authentication (Jupyter):**
- Simple token-based auth for Docker Jupyter containers
  - Token: Configurable in `environment.docker_start_command`
  - Default: Empty token for local development

## Monitoring & Observability

**Error Tracking:**
- None - No external error tracking service (Sentry, etc.)

**Logs:**
- Python logging module via `src/utils/log.py`
  - Console output with Rich formatting
  - Log levels: DEBUG, INFO, WARNING, ERROR
  - No external log aggregation (cloud logging, etc.)

**Cost Tracking:**
- Built-in token cost tracking via mini-swe-agent/litellm
  - Tracks input/output tokens
  - Calculates cost using custom per-token rates
  - Configured in `config/defaults.yaml` models section

## CI/CD & Deployment

**Hosting:**
- Local development - No cloud hosting configured
- Trajectory viewer: Vite dev server on localhost:5173

**CI Pipeline:**
- None - No GitHub Actions, GitLab CI, or similar configured

## Environment Configuration

**Required env vars:**
- `OPENAI_API_KEY` - OpenAI API key (or alternate provider key)
- `ANTHROPIC_API_KEY` - Anthropic API key (if using Claude models)

**Optional env vars:**
- `NOTEBOOK_AGENT_CONFIG_PATH` - Path to config file (default: `./config/agent.yaml`)
- `NOTEBOOK_AGENT_DEFAULTS_PATH` - Path to defaults file (default: `./config/defaults.yaml`)

**Secrets location:**
- `.env` file in project root (gitignored)
- `api_keys.txt` file for multi-threaded execution (gitignored)
- Loaded via python-dotenv in `setup_env.py` and `src/utils/yaml_parser.py`

## Webhooks & Callbacks

**Incoming:**
- None - No webhook endpoints

**Outgoing:**
- None - No external API callbacks (except LLM API calls)

## Docker Integration

**Docker SDK:**
- Python Docker SDK (`docker >= 7.1.0`)
- Functions:
  - Container lifecycle: create, start, stop, remove
  - Volume mounting for notebook files
  - Port mapping for Jupyter server
  - Environment configuration
- Used in: `src/sandbox.py`

**Docker Images:**
- Primary: `yarinamomo/kaggle_python_env` - Python ML environment with Jupyter
- Alternative: `yarinamomo/junobench-simple` - Simplified JunoBench environment
- Source: Docker Hub

## WebSocket Integration

**Jupyter Kernel WebSocket:**
- Protocol: Jupyter messaging protocol over WebSocket
- Client: `websocket-client >= 1.9.0`
- Connection URL: `ws://127.0.0.1:{port}/api/kernels/{kernel_id}/channels`
- Message types handled:
  - `execute_request` - Send code for execution
  - `execute_reply` - Execution result
  - `stream` - stdout/stderr output
  - `execute_result` - Expression result
  - `display_data` - Data display
  - `error` - Execution errors
- Used in: `src/sandbox.py`

## External Libraries (Python)

**mini-swe-agent 2.2.2:**
- Provides SWE agent framework
- Key imports:
  - `minisweagent.agents.default.DefaultAgent` - Base agent class
  - `minisweagent.models.litellm_model.LitellmModel` - LLM model wrapper
  - `minisweagent.exceptions` - Submit, InterruptAgentFlow, FormatError
  - `minisweagent.config.get_config_path` - Config path resolution
- Used throughout: `src/run_agent.py`, `src/ui_agent.py`, `src/notebook_environment.py`

**nbformat 5.10.4+:**
- Jupyter notebook format parsing
- Operations: load, validate, modify, save .ipynb files
- Used in: `src/utils/nbformat_helper.py`, `src/benchmark.py`

**pydantic:**
- Data validation via BaseModel
- Used in: `src/notebook_environment.py` (NotebookEnvironmentConfig)

## Notebooks and Kernel Execution

**IPython Kernel Protocol:**
- Communication via Jupyter kernel messaging
- Execution of Python code cells
- State management across cell executions
- Interrupt and restart capabilities

---

*Integration audit: 2026-03-24*