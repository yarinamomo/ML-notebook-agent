"""
Custom mini-swe-agent Environment for Jupyter Notebook Sandbox.

This module creates an environment adapter that allows mini-swe-agent to work
with the self-defined Docker-based Jupyter notebook sandbox.
"""

import asyncio
import platform
from typing import Any, TypedDict, NotRequired

# Define exceptions for mini-swe-agent v1 compatibility
from minisweagent.exceptions import Submitted
from pydantic import BaseModel


from .benchmark import BenchmarkProblem

class NotebookEnvironmentConfig(BaseModel):
    """Configuration for the notebook environment."""
    sandbox_settings: dict[str, Any]
    source_path: str
    docker_mount_path: str
    problem_mode: str = "JunoBench_Buggy"
    timeout: int = 30

class NotebookEnvironment:
    """mini-swe-agent Environment for Jupyter Notebook Sandbox."""
    def __init__(
        self,
        *,
        config_class: type = NotebookEnvironmentConfig,
        **kwargs
    ):
        """
        Initialize the notebook environment.
        
        Args:
            config_class: Configuration class to use
            **kwargs: Configuration parameters (sandbox_settings, source_path, docker_mount_path, etc.)
        """
        self.config = config_class(**kwargs)
        self.problem: BenchmarkProblem | None = None
        self._setup()
        
    def _setup(self):
        """Setup the benchmark problem and notebook."""
        self.problem = BenchmarkProblem(
            sandbox_settings=self.config.sandbox_settings,
            source_path=self.config.source_path,
            docker_source_path=self.config.docker_mount_path,
            problem_mode=self.config.problem_mode,
            timeout=self.config.timeout
        )
        self.problem.setup()
    
    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        string_command = action.get("command", "")
        output =  self.__execute(string_command)
        if not output.get("exception_info"):
            output["exception_info"] = ""
        return output


    def __execute(self, command: str) -> dict[str, Any]:
        """
        Execute a command (Python code or special notebook operations).
        
        This method is called by mini-swe-agent to execute code. It handles:
        1. Special notebook operation commands (Format: __NOTEBOOK_OP__<operation>(<args>))
        2. Bash commands (wrapped in subprocess)
        3. Regular Python code (executed in notebook kernel)
        
        Args:
            command: Code or operation to execute
            
        Returns:
            dict with keys:
                - output: Combined stdout/stderr/result
                - returncode: 0 for success, 1 for error
        """
        # Strip markdown code blocks if present
        
        # Check if command contains __NOTEBOOK_OP__ - prioritize notebook operations
        # even if it also has bash commands
        if "__NOTEBOOK_OP__" in command:
            # If it's a mixed command (bash + notebook ops), reject it
            if "&&" in command or ";" in command or "|" in command:
                return {
                    "output": (
                        "Error: Cannot mix bash commands with __NOTEBOOK_OP__ commands.\n"
                        "Please execute them separately.\n"
                        f"Your command: {command[:100]}..."
                    ),
                    "returncode": 1,
                    "exception_info": "Mixed bash and notebook operation commands",
                }
            # Pure notebook operation
            if command.strip().startswith("__NOTEBOOK_OP__"):
                command = command.strip().replace("__NOTEBOOK_OP__", "")
                loop = self._get_event_loop()
                return loop.run_until_complete(
                    self.problem.execute_notebook_command(command)
                )
        
        # If it's a bash command, wrap it
        if self._is_bash_command(command):
            command = self._wrap_bash_command(command)
        
        # Execute the bash/Python code in the notebook kernel
        # Execute synchronously by running async in event loop
        loop = self._get_event_loop()            
        result = loop.run_until_complete(
            self.problem.execute_python_command(command)
        )
        
        # Check if task is finished (raises Submitted exception if complete)
        self._check_finished(result)
        
        # Convert result to agent-expected format
        return result
    
    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        """
        Get template variables for mini-swe-agent prompts.
        
        This method is called by mini-swe-agent to get context about the environment
        that can be injected into prompts using {variable_name} syntax.
        
        Returns:
            dict: Template variables including notebook metadata and instructions
        """
        template_vars = {
            # Paths
            "source_path": self.config.source_path,
            "docker_mount_path": self.config.docker_mount_path,
            "problem_mode": self.config.problem_mode,
            "container_working_dir": "/app/container",
            
            # Notebook metadata
            "notebook_initialized": False,
            "cell_count": 0,
        }
        
        # Add notebook-specific info if available
        if self.problem and self.problem.notebook:
            template_vars.update({
                "notebook_initialized": True,
                "cell_count": self.problem.notebook.get_cell_count(),
                
                # Instructions for getting notebook content
                "notebook_access_hint": (
                    "Use __NOTEBOOK_OP__get_cells() to view all notebook cells. "
                    f"The notebook has {self.problem.notebook.get_cell_count()} cells."
                ),
            })
        
        return template_vars
    
    def _check_finished(self, output: dict):
        """
        Check if the output indicates task completion.
        Raises Submitted exception if first line is COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
        and returncode is 0.
        
        Compatible with mini-swe-agent v1.
        """
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission })
    
    def cleanup(self):
        """Cleanup the Docker container and resources."""
        if self.problem:
            self.problem.teardown()
            self.problem = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
    
    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop with Windows compatibility."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def _is_bash_command(self, command: str) -> bool:
        """Check if command looks like a bash command."""
        bash_indicators = [
            'echo ', 'cat ', 'ls ', 'pwd', 'cd ', 'mkdir ', 'rm ', 'touch ',
            'grep ', 'find ', 'sed ', 'awk ', 'git ', 'python ', 'pip ',
            'export ', 'source ', './','bash ', 'sh '
        ]
        command_lower = command.strip().lower()
        return any(command_lower.startswith(indicator) for indicator in bash_indicators)
    
    def _wrap_bash_command(self, command: str) -> str:
        """Wrap bash command in Python subprocess call, matching mini-swe-agent pattern."""
        return f"""
import subprocess

result = subprocess.run(
    {repr(command)},
    shell=True,
    capture_output=True,
    text=True,
    cwd='/app/container',
    encoding='utf-8',
    errors='replace'
)

# Print stdout and stderr
if result.stdout:
    print(result.stdout, end='')
if result.stderr:
    print(result.stderr, end='')

# Print return code marker for parsing
print(f'__RETURNCODE__={{result.returncode}}')
"""
    
    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }