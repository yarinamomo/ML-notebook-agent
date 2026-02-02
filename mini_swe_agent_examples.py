"""
Using mini-swe-agent with the Jupyter Notebook Sandbox
"""

from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model
from src.notebook_environment import NotebookEnvironment


class LoggingModelWrapper:
    """Wrapper that logs all model responses."""
    
    def __init__(self, model, log_file=None, verbose=True):
        self.model = model
        self.log_file = log_file
        self.verbose = verbose
        self.step = 0
    
    def query(self, *args, **kwargs):
        """Intercept query() method - this is what mini-swe-agent actually calls."""
        response = self.model.query(*args, **kwargs)
        self._log_response(response)
        return response
    
    def __call__(self, *args, **kwargs):
        """Intercept __call__ method as fallback."""
        response = self.model(*args, **kwargs)
        self._log_response(response)
        return response
    
    def _log_response(self, response):
        """Log the response."""
        self.step += 1
        
        # Handle dict responses (from litellm)
        if isinstance(response, dict):
            content = response.get('content', '')
            extra = response.get('extra', {})
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"🤖 AGENT RESPONSE (Step {self.step}):")
                print(f"CONTENT (what agent wants to execute):")
                print(content)
                print(f"\nMETADATA:")
                if 'response' in extra:
                    llm_response = extra['response']
                    print(f"  Model: {llm_response.get('model', 'unknown')}")
                    print(f"  Tokens: {llm_response.get('usage', {})}")
                print(f"{'='*60}\n")
            
            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"AGENT RAW RESPONSE - Step {self.step}\n")
                    f.write(f"CONTENT:\n{content}\n\n")
                    f.write(f"FULL RESPONSE:\n{response}\n")
                    f.write(f"{'='*80}\n\n")
        else:
            # String response
            response_str = str(response)
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"🤖 AGENT RESPONSE (Step {self.step}):")
                print(response_str[:1000])
                if len(response_str) > 1000:
                    print(f"\n... (truncated, total length: {len(response_str)} chars)")
                print(f"{'='*60}\n")
            
            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"AGENT RAW RESPONSE - Step {self.step}\n")
                    f.write(response_str)
                    f.write(f"\n{'='*80}\n\n")
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model."""
        return getattr(self.model, name)


def example_with_agent(source_path, problem_mode="JunoBench_Buggy"):
    """Example with custom agent configuration."""
    
    env = NotebookEnvironment(
        sandbox_settings={
            "image_name": "yarinamomo/kaggle_python_env",
            "port": 8888,
        },
        source_path=source_path,
        docker_mount_path="example/docker_mount/",
        problem_mode=problem_mode,
        with_debugger=False,
        log_file="agent_interaction.log",  # Enable logging to file
        verbose=True,  # Set to False to disable console output
    )
    
    try:
        from minisweagent.models.litellm_model import LitellmModel
        
        # Add progress indicator
        print("\n" + "="*80)
        print("🚀 Starting agent...")
        
        # Fast and cheap model for development
        base_model = LitellmModel(
            model_name="gpt-4o-mini"
            )
        
        # Wrap model to log all responses
        model = LoggingModelWrapper(
            base_model, 
            log_file="agent_interaction.log",
            verbose=True
        )
        print("✅ Model created with logging wrapper")
        
        agent = DefaultAgent(
            model=model,
            env=env,
            # Custom configuration
            step_limit=30,  # Max number of steps
            cost_limit=3.0,  # Max cost in dollars
            system_template="""You are a Python-based machine learning Jupyter notebook debugging expert.
Your task is to fix cell CRASHES in a given machine learning Jupyter notebook. Do not care if the cell does not CRASH.

IMPORTANT RULES:
1. Before each action, briefly explain why you're doing it, then call the bash tool with the command
2. Execute ONE OPTION at a time
3. DO NOT chain commands with && or ; or |
4. DO NOT mix __NOTEBOOK_OP__ commands with bash commands
5. Wait for each command to complete before issuing the next one

OPTION 1: NOTEBOOK OPERATIONS:
To interact with the notebook, use the bash tool, with bash keyword, with one of the following commands prefixed with __NOTEBOOK_OP__ :
   __NOTEBOOK_OP__get_cell_count()          # Returns the total number of cells
   __NOTEBOOK_OP__get_cells()               # Returns all cells with their content
   __NOTEBOOK_OP__get_cell(cell_index)               # Get specific cell by index (0-based)
   __NOTEBOOK_OP__edit_cell(cell_index, "new Python code")  # Edit cell at index
   __NOTEBOOK_OP__run_cell(cell_index)               # Run specific cell by index
   __NOTEBOOK_OP__run_all()                 # Run all cells in order
Use bash command, for example:
```bash
__NOTEBOOK_OP__run_all()
```
But NOT:
```
__NOTEBOOK_OP__run_all()
```

OPTION 2: REGULAR PYTHON CODE:
You can also execute regular Python code directly in the notebook kernel by calling the bash tool, for example:
```bash
import pandas as pd
df = pd.read_csv('/app/container/data/train.csv')
print(df.head())
```

OPTION 3: When done, call bash tool with:
```bash
echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
```

WORKFLOW EXAMPLE (execute ONE command per step using the bash tool):
Step 1: Call bash tool with: __NOTEBOOK_OP__get_cells()
Step 2: Call bash tool with: __NOTEBOOK_OP__run_all()
Step 3: (if errors found) analyze the error by writing Python code to run diagnoses, or trying to fix the code by calling bash tool with: __NOTEBOOK_OP__edit_cell(cell_index, "fixed Pythoncode")
Step 4: check if the fix works by calling bash tool with: __NOTEBOOK_OP__run_cell(cell_index) or __NOTEBOOK_OP__run_all()
Step 5: if more errors, repeat Steps 3-4 until all cells run successfully
Step 6: if no errors remain, terminate by calling bash tool with: echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
```
""",
        )
        print("✅ Agent created")
        print("🏃 Running agent with task...")
        
        import time
        import sys
        start_time = time.time()
        
        # Flush output to ensure we see progress
        sys.stdout.flush()
        
        try:
            exit_status, result = agent.run(
                "Run all the cells of the notebook in order and make sure no cell crashes, fix any errors you find."
            )
            
            elapsed = time.time() - start_time
            print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
            print(f"Exit Status: {exit_status}")
            print(f"Result:\n{result}")
            
            # # Log full agent trajectory if available
            # if hasattr(agent, 'trajectory') and env.log_file:
            #     with open(env.log_file, 'a', encoding='utf-8') as f:
            #         f.write("\n" + "="*80 + "\n")
            #         f.write("FULL AGENT TRAJECTORY\n")
            #         f.write("="*80 + "\n")
            #         for i, step in enumerate(agent.trajectory):
            #             f.write(f"\n--- Step {i+1} ---\n")
            #             if hasattr(step, 'thought'):
            #                 f.write(f"THOUGHT: {step.thought}\n")
            #             if hasattr(step, 'action'):
            #                 f.write(f"ACTION: {step.action}\n")
            #             if hasattr(step, 'observation'):
            #                 f.write(f"OBSERVATION: {step.observation}\n")
            #             f.write("\n")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Agent failed after {elapsed:.1f} seconds")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    finally:
        env.cleanup()
        
        # Show log file location
        print(f"\n📝 Full interaction log saved to: agent_interaction.log")


def example_with_tools(source_path, problem_mode="JunoBench_Buggy"):
    """Example showing how to manually use the environment (without agent)."""
    
    env = NotebookEnvironment(
        sandbox_settings={
            "image_name": "yarinamomo/kaggle_python_env",
            "port": 8888,
        },
        source_path=source_path,
        docker_mount_path="example/docker_mount/",
        problem_mode=problem_mode,
        with_debugger=False,
        log_file="agent_interaction.log",  # Enable logging to file
        verbose=True,  # Set to False to disable console output
    )
    
    try:
        notebook = env.notebook
        print(f"   Notebook has {notebook.get_cell_count()} cells")
        
    finally:
        env.cleanup()

# def example_interactive():
#     """Example with human-in-the-loop interaction."""
#     from minisweagent.agents.interactive import InteractiveAgent
    
#     env = NotebookEnvironment(
#         sandbox_settings={
#             "image_name": "yarinamomo/kaggle_python_env",
#             "port": 8888,
#         },
#         source_path="example/test/",
#         docker_mount_path="example/docker_mount/"
#     )
    
#     try:
#         # Interactive agent lets you approve each action
#         agent = InteractiveAgent(
#             model=get_model("gpt-5"),
#             env=env,
#         )
        
#         exit_status, result = agent.run(
#             "Fix notebook errors. Show me each fix before applying it."
#         )
        
#         print(f"Exit Status: {exit_status}")
#         print(f"Result:\n{result}")
#     finally:
#         env.cleanup()
