# Architecture

**Analysis Date:** 2026-03-24

## Pattern Overview

**Overall:** Agent-Sandbox-Orchestration pattern

**Key Characteristics:**
- AI-agent-driven notebook debugging via structure tool calling
- Docker-based isolated execution environment with WebSocket communication
- Dual execution modes: iterative agent and single-shot baseline
- Layered configuration system (defaults + overlay)
- Trajectory logging and analysis pipeline

## Layers

**Agent Layer:**
- Purpose: Provides AI-driven debugging with tool-based interaction
- Location: `src/`
- Contains: `ui_agent.py`, `CustomToolLitellmModel.py`, `notebook_tools.py`
- Depends on: mini-swe-agent framework, LLM providers
- Used by: Runner layer

**Environment Layer:**
- Purpose: Adapts Jupyter sandbox for mini-swe-agent environment interface
- Location: `src/notebook_environment.py`
- Contains: Tool dispatch logic, state management, submission handling
- Depends on: Benchmark problem layer, mini-swe-agent exceptions
- Used by: Agent layer

**Sandbox Layer:**
- Purpose: Docker container lifecycle and Jupyter kernel communication
- Location: `src/sandbox.py`
- Contains: Container management, WebSocket messaging, kernel control
- Depends on: docker, websocket-client, requests
- Used by: Benchmark problem layer

**Benchmark Problem Layer:**
- Purpose: Problem definition and notebook cell state management
- Location: `src/benchmark.py`
- Contains: Notebook loading, cell editing, execution tracking
- Depends on: nbformat, Sandbox layer
- Used by: Environment layer

**Runner Layer:**
- Purpose: Orchestrates single instance execution (agent or baseline)
- Location: `src/run_agent.py`, `src/run_baseline.py`
- Contains: Instance execution, trajectory/summary saving
- Depends on: All lower layers
- Used by: CLI orchestration

**CLI Orchestration Layer:**
- Purpose: Multi-instance execution with threading and progress tracking
- Location: `main.py`
- Contains: Config loading, instance discovery, progress management
- Depends on: Runner layer, yaml_parser
- Used by: End users

## Data Flow

**Agent Mode Execution:**

1. Configuration loading layered from defaults + overlay YAML files
2. Docker container started with Jupyter server
3. WebSocket connection established to Jupyter kernel
4. Agent loop begins:
   - LLM queries with notebook tools exposed (get_cells, edit_cell, run_all, etc.)
   - LLM returns structured tool calls
   - Environment validates and dispatches to handlers
   - Sandbox sends execute requests via WebSocket
   - Kernel processes and returns messages
   - WebSocket thread accumulates outputs (stream, error, execute_result)
   - Results formatted and fed back to LLM
5. Loop until submit() called or timeout/cost limit reached
6. Trajectory (messages) and summary written to disk
7. Docker container and kernel cleaned up

**Baseline Mode Execution:**

1. Notebook cells collected and formatted
2. All cells run once to collect error outputs
3. Full notebook + errors sent to LLM in single prompt
4. LLM returns multiple edit_cell calls to fix all errors
5. Edits applied to cells
6. All cells run again to verify fixes
7. Status determined, results saved

**State Management:**
- Kernel state managed by Jupyter within Docker container
- Cell state tracked (edited/unchanged) via `_cell_states` dict
- Execution results cached in `_exec_states` dict
- Trajectory captures complete message history (LLM responses, tool calls, observations)

## Key Abstractions

**Structured Tool Calling:**
- Purpose: Replaces string-based bash commands with typed function calls
- Examples: `src/notebook_tools.py` (EDIT_CELL_TOOL, RUN_ALL_TOOL, etc.)
- Pattern: OpenAI function-calling schema with parameters, descriptions, required fields
- Fallback: Content parsing for models that embed tool calls in text (e.g., GLM)

**Notebook Environment:**
- Purpose: Mini-swe-agent environment adapter for Jupyter debugging
- Examples: `NotebookEnvironment` class in `src/notebook_environment.py`
- Pattern: Tool dispatch via match statement, exception flow control via Submitted

**Docker Sandbox:**
- Purpose: Isolated kernel execution with timeout and restart capability
- Examples: `DockerSandbox` class in `src/sandbox.py`
- Pattern: REST API for kernel lifecycle, WebSocket for execution, polling for completion

**Configuration Layering:**
- Purpose: Separate shared defaults from run-specific overrides
- Examples: `config/defaults.yaml` + `config/agent.yaml`
- Pattern: Deep merge where override values take precedence

## Entry Points

**`main.py`:**
- Location: `main.py`
- Triggers: CLI command execution
- Responsibilities:
  - Load and merge configuration layer (defaults + overlay)
  - Discover instances or use single target
  - Build execution queue (instance × run number combinations)
  - Support optional multi-threaded execution with worker pools
  - Track progress across all instances
  - Retry failed runs
  - Report summary of incomplete instances

**Preprocessing:**
- Location: `preprocess_trajectories.py`
- Triggers: Manual invocation for trajectory viewer
- Responsibilities: Read trajectory JSON files, produce `viewer/public/data.json`

**Data Analysis:**
- Location: `data_analysis/*.py`
- Triggers: Manual invocation for analysis
- Responsibilities: Extract patterns from trajectories, generate reports

## Error Handling

**Strategy:** Multi-level with flow control exceptions

**Patterns:**
- **Agent flow interrupts:** `InterruptAgentFlow` and subclasses (`Submitted`, `AgentTimeout`, `EnvironmentUnavailable`) for controlled exits
- **Timeout handling:** Kernel interrupt via REST API, automatic restart, retry logic
- **WebSocket disconnection:** Automatic reconnection with exponential backoff, manual verification
- **Tool format errors:** `FormatError` with template-based error messages fed to LLM
- **Execution failures:** Return error outputs to agent for diagnosis

## Cross-Cutting Concerns

**Logging:** Structured logging via `src/utils/log.py` with rich progress bars
**Validation:** Configuration validation via Pydantic models (`NotebookEnvironmentConfig`)
**Authentication:** API keys loaded from .env or provided via threading parameters
**Timeout Protection:** Per-execution timeout (default 30s), total timeout (default 3600s), cost limits
**Trajectory Tracking:** Complete message history saved per instance for analysis

---

*Architecture analysis: 2026-03-24*