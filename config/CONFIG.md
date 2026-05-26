# YAML Configuration Guide

Our LLM-based notebook repair agent builds out of the `mini-swe-agent` framework, adding custom parameters specific to our Jupyter notebook environments and execution patterns. We use a two-layer configuration system to keep execution profiles clean while reusing standard environment or agent settings.

For standard settings (like `commands`, `history_processor`, `demonstrations`), refer to the underlying `mini-swe-agent` documentation:
[mini-swe-agent YAML configuration documentation](https://mini-swe-agent.com/latest/advanced/yaml_configuration/)

This document explains our configuration system, the hierarchical layering, and the parameters specifically extended for this project.

## Configuration Mechanics

The execution is governed by merging two YAML files:
1. **Defaults config:** Provides overarching parameters (models available, environment timeouts, default run mode). Controlled by `NOTEBOOK_AGENT_DEFAULTS_PATH` in `.env`.
2. **Run-specific (overlay) config:** Overrides specific fields for a concrete experiment (e.g., whether to output cell results or change the system prompt). Controlled by `--config` in the CLI or `NOTEBOOK_AGENT_CONFIG_PATH` in `.env`.

When you run `python main.py --config config/agent.yaml`, it executes [`yaml_parser.load_config(config_spec, defaults_spec)`](../src/utils/yaml_parser.py), which recursively merges the overlay into the defaults using `_deep_merge()`.

## Extended Configuration Sections

Below are the sections and options customized or added specifically for this project (on top of `mini-swe-agent` defaults):

### 1. `environment` settings (Custom)
This block is entirely specific to our environment. It provides the settings needed to spin up the Jupyter kernel sandbox and isolate the evaluation.

```yaml
environment:
  docker_image_name: yarinamomo/kaggle_python_env  # Dedicated Docker image for Jupyter isolation
  problem_mode: JunoBench_Buggy                    # Legacy parameter (do not change). Ensures the parser correctly reads the notebook format.
  source_path_parent: example/test_case/           # Location of benchmark files (contains a subdirectory for each instance)
  run_all_instances: true                          # If true, runs all instances found within source_path_parent.
  target_nb_instance: sklearn_1                    # Specify single instance directory ID (used if run_all_instances is false)
  docker_mount_path: example/docker_mount/         # Temporary mount path for docker execution
  docker_start_command: "jupyter server --allow-root ..." # Bootstraps the local jupyter instance
  timeout: 1800                                    # Overall environment timeout 
  no_runtime_output: false                         # Controls whether tool calls omit Jupyter output
```

### 2. `misc` settings (Custom)
This is an entirely custom block added by our framework. It controls meta-execution, the agent mode, outputs, and threading.

```yaml
misc:
  mode: agent                         # Defines the runner style: 'agent' or 'baseline'
  run_count: 3                        # Number of successive trials/repeats to execute
  if_local_key: true                  # Flag to load keys from .env directly via dotenv overrides
  skip_existing: true                 # Flag to skip files that already have a successful run completion summary
  trajectory_log_path: ./trajectories/agent # Output directory for agent trajectories (.traj.json logs)
```

### 3. `models` list handling (Custom Extension)
By default, `mini-swe-agent` utilizes a single `model:` block to define the LLM. While we still support providing a single `model` fallback (see [`main.py`](../main.py)), our configuration extends this by leveraging a **`models` list** (plural) directly in the defaults file:

```yaml
models:
  - model_name: glm-4.7-355b
    cost_tracking: ignore_errors
    model_kwargs:
      custom_llm_provider: openai
      api_base: https://aqueduct.ai.datalab.tuwien.ac.at/v1
      input_cost_per_token: 0.00000027778
      output_cost_per_token: 0.00000111111
  # Add other models here... 
```

`main.py` dynamically selects a model from this list (using the `-m` / `--model` CLI argument or falling back to the first one available) and automatically maps it to the standard `model` key via [`src.utils.yaml_parser.set_model_config`](../src/utils/yaml_parser.py).

### 4. `agent` config block (`mini-swe-agent` Standard)
The structural block for `agent` and its underlying formatting hooks are completely standard to the `mini-swe-agent` framework. However, we customize the actual template values heavily out-of-the-box (e.g. in [`agent.yaml`](agent.yaml)) to specifically guide the framework toward diagnosing and repairing Jupyter notebooks instead of generic software engineering tasks:

- **`system_template`**: Guides the agent on fixing code errors, dealing with crash evaluation, and executing cell refreshes (`run_all`, `run_cell`). 
- **`instance_template`**: Feeds the agent the actual crashing notebook and outputs.
- **`action_observation_template`**: Specific formatting hook customized to truncate cell output longer than 20,000 characters and inform the agent to use diagnosis tools to avoid context bounds problems.