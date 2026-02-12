"""
Custom mini-swe-agent Environment for Jupyter Notebook Sandbox.

This module creates an environment adapter that allows mini-swe-agent to work
with the self-defined Docker-based Jupyter notebook sandbox.
"""

import asyncio
from dataclasses import dataclass
import json
from typing import Any, Optional

# Define exceptions for mini-swe-agent v1 compatibility
from minisweagent.agents.default import Submitted
from src.utils.nb_types import CellExecutionResult

from .benchmark import BenchmarkProblem
from .utils.notebook_command_helper import (
    NotebookCommandType,
    is_bash_command,
    is_mixed_notebook_and_bash,
    is_notebook_command,
    parse_notebook_command,
    strip_markdown_code_blocks,
    wrap_bash_command,
)

@dataclass
class NotebookEnvironmentConfig:
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
        self.problem: BenchmarkProblem = BenchmarkProblem(
            sandbox_settings=self.config.sandbox_settings,
            source_path=self.config.source_path,
            docker_source_path=self.config.docker_mount_path,
            problem_mode=self.config.problem_mode,
            timeout=self.config.timeout
        )        
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """
        Execute a command (Python code or special notebook operations).
        
        This method is called by mini-swe-agent to execute code. It handles:
        1. Special notebook operation commands (Format: __NOTEBOOK_OP__<operation>(<args>))
        2. Bash commands (wrapped in subprocess)
        3. Regular Python code (executed in notebook kernel)
        
        Args:
            command: Code or operation to execute
            cwd: Working directory
            timeout: Execution timeout
            
        Returns:
            dict with keys:
                - output: Combined stdout/stderr/result
                - returncode: 0 for success, 1 for error
        """
        # Strip markdown code blocks if present
        command = strip_markdown_code_blocks(command)
        

        if is_mixed_notebook_and_bash(command):
            return self._wrap_error(
                "Error: Cannot mix bash commands with __NOTEBOOK_OP__ commands.\n"
                "Please execute them separately.\n"
                "Your command: {command[:100]}...", 
                "Mixed bash and notebook operation commands")
        exec_result = None # TODO this is a bit hacky. We need to have access to the execution result in _check_finished, but it's only produced in certain branches. Refactor needed.
        try:
            if is_notebook_command(command):
                return self._execute_notebook_command(command.strip().replace("__NOTEBOOK_OP__", "", 1))
            # If it's a bash command, wrap it
            elif is_bash_command(command):
                command = wrap_bash_command(command)            
                exec_result = self.problem.execute_python_command(command)
                result = self._wrap_success(self._format_exec_result(exec_result))
            else:
                result = self._wrap_error("Unrecognized command format. Please use __NOTEBOOK_OP__ for notebook operations or valid bash commands.")
        except Exception as exc:
            result = self._wrap_error(f"Error executing command: {exc}")

        # Check if task is finished (raises Submitted exception if complete)
        self._check_finished(exec_result)

        # Convert result to agent-expected format
        return result
    
    def get_template_vars(self) -> dict[str, Any]:
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
        if self.problem:
            template_vars.update({
                "notebook_initialized": True,
                "cell_count": self.problem.get_cell_count(),
                
                # Instructions for getting notebook content
                "notebook_access_hint": (
                    "Use __NOTEBOOK_OP__get_cells() to view all notebook cells. "
                    f"The notebook has {self.problem.get_cell_count()} cells."
                ),
            })
        
        return template_vars
    
    def _check_finished(self, exec_result: Optional[CellExecutionResult]):
        """
        Check if the output indicates task completion.
        Raises Submitted exception if first line is COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
        and returncode is 0.
        
        Compatible with mini-swe-agent v1.
        """
        outputs = exec_result.get("outputs", []) if exec_result else []
        text = outputs[0].get("text", "") if outputs else ""
        if ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in text 
            and "RETURNCODE=0" in text):
            raise Submitted(text)

    def _execute_notebook_command(self, command: str) -> dict[str, Any]:
        if not self.problem.get_cell_count():
            return self._wrap_error("Notebook not initialized")
        try:
            parsed = parse_notebook_command(command)
            match parsed.kind:
                case NotebookCommandType.GET_CELL_COUNT:
                    return self._wrap_success(str(self.problem.get_cell_count()))
                case NotebookCommandType.GET_CELLS:
                    cells = self.problem.get_cells()
                    return self._wrap_success("\n\n".join(cells))
                case NotebookCommandType.GET_CELL:
                    index = parsed.args[0]
                    return self._wrap_success(self.problem.get_cell(index))
                case NotebookCommandType.EDIT_CELL:
                    index, code = parsed.args
                    self.problem.edit_cell(index, code)
                    return self._wrap_success(f"Cell {index} edited successfully")
                case NotebookCommandType.RUN_CELL:
                    index = parsed.args[0]
                    exec_result = self.problem.run_cell(index)
                    return self._wrap_success(self._format_exec_result(exec_result))
                case NotebookCommandType.RUN_ALL:
                    exec_results = self.problem.run_all()
                    output_parts = []
                    for i, exec_result in enumerate(exec_results):
                        cell_output = self._format_exec_result(exec_result)
                        output_parts.append(f"Cell {i}:\n{cell_output}")
                    return self._wrap_success("\n\n".join(output_parts))
        except Exception as exc:
            return self._wrap_error(f"Error executing notebook command: {exc}")

    def _wrap_success(self, output: str) -> dict[str, Any]:
        return {
            "output": output,
            "returncode": 0,
            "exception_info": "",
        }

    def _wrap_error(self, message: str, exception_info: Optional[str] = None) -> dict[str, Any]:
        return {
            "output": message,
            "returncode": 1,
            "exception_info": exception_info or message,
        }

    def _format_exec_result(self, exec_result: Any) -> str:
        if exec_result is None:
            return "(No output)"
        try:
            return json.dumps(exec_result)
        except TypeError:
            return str(exec_result)
    
    def close(self):
        """Cleanup the Docker container and resources."""
        if self.problem:
            self.problem.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
