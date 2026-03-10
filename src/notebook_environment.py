"""
Custom mini-swe-agent Environment for Jupyter Notebook Sandbox.

This module creates an environment adapter that allows mini-swe-agent to work
with the self-defined Docker-based Jupyter notebook sandbox.
"""

from dataclasses import dataclass
from typing import Any, Optional, TypedDict, Literal, cast
from pathlib import Path
from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from src.utils.nb_types import CellExecutionResult, ErrorOutput
from src.utils.format_nb_cells import format_exec_result_for_llm, format_cell_source_for_llm, format_initial_notebook

from .benchmark import BenchmarkProblem
from src.utils.log import logger

class EnvironmentResult(TypedDict):
    output: str
    returncode: Literal[0, 1]
    exception_info: Optional[str]

class NotebookEnvironmentConfig(BaseModel):
    """Configuration for the notebook environment."""
    source_path_parent: str = "example/JunoBench/"
    docker_image_name: str = "yarinamomo/junobench-simple"
    port: int = 8888
    docker_start_command: str | None = None
    docker_mount_path: str =  "example/docker_mount/"
    problem_mode: str = "JunoBench_Buggy"
    timeout: int = 600
    no_runtime_output: bool = False

class NotebookEnvironment:
    """mini-swe-agent Environment for Jupyter Notebook Sandbox."""
    def __init__(
        self,
        instance_name: str,
        output_dir: Path,
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
        self.source_path = Path(self.config.source_path_parent) / instance_name
        self.is_submittable = False # Flag to track if submission is allowed (after successful run_all)
        self.problem: BenchmarkProblem = BenchmarkProblem(
            sandbox_settings={
                "image_name": self.config.docker_image_name,
                "port": self.config.port,
                "start_command": self.config.docker_start_command,
            },
            source_path=self.source_path,
            output_dir=output_dir,
            problem_mode=self.config.problem_mode,
            docker_source_path=self.config.docker_mount_path,
            timeout=self.config.timeout,
        )        
    
    def execute(self, action: dict, cwd: str = "") -> EnvironmentResult:
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
            logger.exception(f"Error executing tool '{tool_name}'")
            logger.exception(exc)
            return self._wrap_error(f"Error executing {tool_name}: {exc}")

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, tool_name: str, args: dict) -> EnvironmentResult:
        """Route a structured tool call to the appropriate handler."""
        if not self.problem.get_cell_count() and tool_name not in ("submit",):
            return self._wrap_error("Notebook not initialized")
        tmp_is_submittable = self.is_submittable
        self.is_submittable = self.is_submittable and tool_name != "edit_cell"  # Reset submittable flag on edit_cell
        match tool_name:
            case "get_cell_count":
                return self._wrap_success(str(self.problem.get_cell_count()))

            case "get_cells":
                cells = self.problem.get_cells()
                cell_sources = [format_cell_source_for_llm(i, cell) for i, cell in enumerate(cells)]
                return self._wrap_success("\n\n".join(cell_sources))

            case "get_cell":
                index = int(args["cell_index"])
                cell_source = format_cell_source_for_llm(index, self.problem.get_cell(index))
                return self._wrap_success(cell_source)

            case "edit_cell":
                index = int(args["cell_index"])
                code = str(args["code"])
                self.problem.edit_cell(index, code)
                return self._wrap_success(f"Cell {index} edited successfully")

            case "run_cell":
                index = int(args["cell_index"])
                exec_result = self.problem.run_cell(index)
                return self._wrap_execution_result(exec_result)

            case "run_all":
                exec_results = self.problem.run_all()
                last_exec_result = exec_results[-1] if exec_results else None
                result = self._wrap_execution_result(last_exec_result)
                output_parts = []
                for i, exec_result in enumerate(exec_results):
                    cell_output = format_exec_result_for_llm(exec_result, if_truncate=True, no_runtime_output=self.config.no_runtime_output)
                    output_parts.append(f"Cell {i}:\n{cell_output}")

                result["output"] = "\n\n".join(output_parts)
                self.is_submittable = result["returncode"] == 0
                return result
            case "run_code":
                code = str(args["code"])
                exec_result = self.problem.execute_python_command(code)
                return self._wrap_execution_result(exec_result)

            case "submit":
                summary = args.get("summary", "")
                raise Submitted({
                    "role": "exit",
                    "content": summary,
                    "extra": {
                        "exit_status": "Submitted" if tmp_is_submittable else "SubmittedWithErrors",
                        "submission": summary,
                    },
                })
            case _:
                return self._wrap_error(
                    f"Unknown tool '{tool_name}'. "
                    "Valid tools: get_cell_count, get_cells, get_cell, "
                    "edit_cell, run_cell, run_all, run_code, submit."
                )


    def get_initial_notebook(self) -> str:
        """Get the original notebook content."""
        return format_initial_notebook(self.problem.get_initial_notebook(), no_runtime_output=self.config.no_runtime_output)

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

    def _wrap_success(self, output: str) -> EnvironmentResult:
        return {
            "output": output,
            "returncode": 0,
            "exception_info": "",
        }

    def _wrap_error(self, message: str, exception_info: Optional[str] = None) -> EnvironmentResult:
        return {
            "output": message,
            "returncode": 1,
            "exception_info": exception_info or message,
        }

    def _wrap_execution_result(self, exec_result: CellExecutionResult | None) -> EnvironmentResult:
        """Wrap a cell execution result into the standard tool output format."""
        formatted_output = format_exec_result_for_llm(exec_result, if_truncate=True, no_runtime_output=self.config.no_runtime_output)
        error = self._get_execution_error(exec_result)
        if error is not None:
            self.is_submittable = False  # Mark as not submittable if there's an execution error
            return self._wrap_error(formatted_output, exception_info=str(error))
        return self._wrap_success(formatted_output)

    def _get_execution_error(self, exec_result: CellExecutionResult | None) -> Optional[ErrorOutput]:
        """Check if execution result contains errors."""
        if exec_result is None or not isinstance(exec_result, dict):
            return None
                
        # Check for error outputs
        outputs = exec_result.get('outputs', [])
        for out in outputs:
            msg_type = out.get('output_type', '')
            if msg_type == 'error':
                return cast(ErrorOutput, out)
        return None
    
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

