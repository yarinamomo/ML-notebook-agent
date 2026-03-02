"""
Custom mini-swe-agent Environment for Jupyter Notebook Sandbox.

This module creates an environment adapter that allows mini-swe-agent to work
with the self-defined Docker-based Jupyter notebook sandbox.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from src.utils.nb_types import CellExecutionResult, format_for_llm

from .benchmark import BenchmarkProblem

class NotebookEnvironmentConfig(BaseModel):
    """Configuration for the notebook environment."""
    source_path_parent: str = "example/JunoBench/"
    docker_image_name: str = "yarinamomo/junobench-simple"
    port : int = 8888
    docker_start_command: str | None = None
    docker_mount_path: str =  "example/docker_mount/"
    problem_mode: str = "JunoBench_Buggy"
    timeout: int = 600
    run_all_timeout: int = 0  # Total timeout for run_all() operation


class NotebookEnvironment:
    """mini-swe-agent Environment for Jupyter Notebook Sandbox."""
    def __init__(
        self,
        source_path: str,
        output_dir: str,
        config_class: type = NotebookEnvironmentConfig,
        **kwargs
    ):
        """
        Initialize the notebook environment.
        
        Args:
            config_class: Configuration class to use
            **kwargs: Configuration parameters (sandbox_settings, source_path, docker_mount_path, etc.)
        """
        self.config: NotebookEnvironmentConfig = config_class(**kwargs)
        self.source_path = source_path
        self.problem: BenchmarkProblem = BenchmarkProblem(
            sandbox_settings={
                "image_name": self.config.docker_image_name,
                "port": self.config.port,
                "start_command": self.config.docker_start_command,
            },
            source_path=source_path,
            output_dir=output_dir,
            problem_mode=self.config.problem_mode,
            docker_source_path=self.config.docker_mount_path,
            timeout=self.config.timeout,
            run_all_timeout=self.config.run_all_timeout
        )        
    
    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        """Execute a structured tool action.

        The *action* dict is produced by ``parse_notebook_tool_actions`` and
        contains ``tool_name``, ``arguments`` and ``tool_call_id``.

        Legacy string-based commands (``action["command"]``) are no longer
        supported; every operation is dispatched by tool name.
        """
        tool_name: str = action.get("tool_name", "")
        args: dict = action.get("arguments", {})

        if not tool_name:
            return self._wrap_error(
                "No tool_name in action. This is a bug – please report it."
            )

        try:
            return self._dispatch_tool(tool_name, args)
        except Submitted:
            raise  # Let the agent framework handle submission
        except Exception as exc:
            return self._wrap_error(f"Error executing {tool_name}: {exc}")

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, tool_name: str, args: dict) -> dict[str, Any]:
        """Route a structured tool call to the appropriate handler."""
        if not self.problem.get_cell_count() and tool_name not in ("submit",):
            return self._wrap_error("Notebook not initialized")

        match tool_name:
            case "get_cell_count":
                return self._wrap_success(str(self.problem.get_cell_count()))

            case "get_cells":
                cells = self.problem.get_cells()
                return self._wrap_success("\n\n".join(cells))

            case "get_cell":
                index = int(args["cell_index"])
                return self._wrap_success(self.problem.get_cell(index))

            case "edit_cell":
                index = int(args["cell_index"])
                code = str(args["code"])
                self.problem.edit_cell(index, code)
                return self._wrap_success(f"Cell {index} edited successfully")

            case "run_cell":
                index = int(args["cell_index"])
                exec_result = self.problem.run_cell(index)
                return self._wrap_success(self._format_exec_result(exec_result))

            case "run_all":
                exec_results = self.problem.run_all()
                output_parts = []
                for i, exec_result in enumerate(exec_results):
                    cell_output = self._format_exec_result(exec_result)
                    output_parts.append(f"Cell {i}:\n{cell_output}")
                return self._wrap_success("\n\n".join(output_parts))

            case "run_code":
                code = str(args["code"])
                exec_result = self.problem.execute_python_command(code)
                return self._wrap_success(self._format_exec_result(exec_result))

            case "submit":
                summary = args.get("summary", "")
                raise Submitted({
                    "role": "exit",
                    "content": summary,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": summary,
                    },
                })
            case _:
                return self._wrap_error(
                    f"Unknown tool '{tool_name}'. "
                    "Valid tools: get_cell_count, get_cells, get_cell, "
                    "edit_cell, run_cell, run_all, run_code, submit."
                )


    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        """
        Get template variables for mini-swe-agent prompts.
        
        Returns:
            dict: Template variables including notebook metadata and instructions
        """
        template_vars = {
            # Paths
            "source_path": self.source_path,
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
                "notebook_access_hint": (
                    "Use the get_cells tool to view all notebook cells. "
                    f"The notebook has {self.problem.get_cell_count()} cells."
                ),
            })
        
        return template_vars

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

    def _format_exec_result(self, exec_result: CellExecutionResult) -> str:
        if exec_result is None:
            return "(No output)"
        if isinstance(exec_result, dict) and 'outputs' in exec_result:
            return format_for_llm(exec_result, if_truncate=True, max_words=500)
        return str(exec_result)
    
    def close(self):
        """Cleanup the Docker container and resources."""
        if hasattr(self, "problem") and self.problem:
            self.problem.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

