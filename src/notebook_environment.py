"""
Custom mini-swe-agent Environment for Jupyter Notebook Sandbox.

This module creates an environment adapter that allows mini-swe-agent to work
with the self-defined Docker-based Jupyter notebook sandbox.
"""

import asyncio
from dataclasses import dataclass
import platform
from typing import Any

# Define exceptions for mini-swe-agent v1 compatibility
from minisweagent.agents.default import Submitted

from .benchmark import BenchmarkProblem, LightweightNotebook
from .sandbox import ExecutionStatus, ExecutionResult, SandboxResultType
import re

@dataclass
class NotebookEnvironmentConfig:
    """Configuration for the notebook environment."""
    sandbox_settings: dict[str, Any]
    source_path: str
    docker_mount_path: str
    problem_mode: str = "JunoBench_Buggy"
    timeout: int = 30
    with_debugger: bool = False
    verbose: bool = True  # Print execution details to console


def _clean_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def _result_parts_to_string(parts: list) -> str:
    """Convert list of ResultPart to clean string output."""
    text_parts = []
    for part in parts:
        if part.type == SandboxResultType.TEXT:
            text_parts.append(_clean_ansi_codes(part.content))
    return ''.join(text_parts)

def _get_status_label(exec_result: ExecutionResult) -> str:
    """Get human-readable status label from ExecutionResult."""
    if exec_result.status == ExecutionStatus.ERROR:
        return "ERROR"
    elif exec_result.status == ExecutionStatus.COMPLETED:
        if exec_result.result and exec_result.result.text.strip():
            return "SUCCESS"
        else:
            return "SUCCESS (no output)"
    elif exec_result.status == ExecutionStatus.CANCELLED:
        return "CANCELLED"
    else:
        return exec_result.status.value.upper()

class NotebookEnvironment:
    """mini-swe-agent Environment for Jupyter Notebook Sandbox."""
    def __init__(
        self,
        *,
        config_class: type = NotebookEnvironmentConfig,
        log_file: str | None = None,
        **kwargs
    ):
        """
        Initialize the notebook environment.
        
        Args:
            config_class: Configuration class to use
            log_file: Optional path to log file for recording agent interactions
            **kwargs: Configuration parameters (sandbox_settings, source_path, docker_mount_path, etc.)
        """
        self.config = config_class(**kwargs)
        self.problem: BenchmarkProblem | None = None
        self.notebook: LightweightNotebook | None = None
        self.log_file = log_file
        self.step_count = 0
        self._setup()
        
        # Initialize log file
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("AGENT INTERACTION LOG\n")
                f.write("="*80 + "\n\n")
        
    def _setup(self):
        """Setup the benchmark problem and notebook."""
        self.problem = BenchmarkProblem(
            sandbox_settings=self.config.sandbox_settings,
            source_path=self.config.source_path,
            docker_source_path=self.config.docker_mount_path,
            problem_mode=self.config.problem_mode
        )
        self.problem.setup(with_debugger=self.config.with_debugger)
        self.notebook = self.problem.notebook
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """
        Execute a command (Python code or special notebook operations).
        
        This method is called by mini-swe-agent to execute code. It handles:
        1. Special notebook operation commands (prefixed with __NOTEBOOK_OP__)
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
        command = command.strip()
        
        # Look for code blocks anywhere in the text (not just at the start)
        if "```" in command:
            # Find the first code block
            lines = command.split('\n')
            code_start = -1
            code_end = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith("```") and code_start == -1:
                    code_start = i
                elif line.strip() == "```" and code_start != -1:
                    code_end = i
                    break
            
            # Extract code between the markers
            if code_start != -1 and code_end != -1 and code_end > code_start:
                # Get lines between ``` markers, excluding the markers themselves
                command = '\n'.join(lines[code_start + 1:code_end])
        
        # Log what the agent is executing
        if self.config.verbose:
            self.step_count += 1
            
            log_entry = f"\n{'='*60}\n"
            log_entry += f"STEP {self.step_count}\n"
            log_entry += f"{'='*60}\n"
            log_entry += f"COMMAND (after stripping code blocks):\n{command}\n"
            log_entry += f"{'='*60}\n"
        
        
            print(f"\n{'='*60}")
            print(f"🤖 AGENT EXECUTING (Step {self.step_count}):")
            print(command[:500])  # Show first 500 chars
            if len(command) > 500:
                print(f"... (truncated, total length: {len(command)} chars)")
            print(f"{'='*60}\n")
        
        if not self.notebook:
            return {
                "output": "Error: Notebook not initialized",
                "returncode": 1
            }
        
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
                    "returncode": 1
                }
            # Pure notebook operation
            if command.strip().startswith("__NOTEBOOK_OP__"):
                return self._handle_notebook_operation(command)
        
        # Check if this is a notebook operation command
        if command.strip().startswith("__NOTEBOOK_OP__"):
            return self._handle_notebook_operation(command)
        
        # If it's a bash command, wrap it
        if self._is_bash_command(command):
            command = self._wrap_bash_command(command)
        
        # Execute the Python code in the notebook kernel
        try:
            # Execute synchronously by running async in event loop
            if platform.system() == 'Windows':
                # Windows may need a new event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            else:
                loop = asyncio.get_event_loop()
            
            exec_result = loop.run_until_complete(
                self.notebook.sandbox.run_async(command, cancel_event=None)
            )
            
            # Extract output as clean string from ExecutionResult
            if exec_result.result:
                parts = exec_result.result.llm_compatible()
                output_str = _result_parts_to_string(parts)
            else:
                output_str = "(No output)"
            
            # Check for returncode marker from bash wrapper and extract it
            returncode = 0 if exec_result.status == ExecutionStatus.COMPLETED else 1
            # Look for the returncode marker pattern
            returncode_match = re.search(r'__RETURNCODE__=(\d+)', output_str)
            if returncode_match:
                # Extract the actual return code from bash subprocess
                returncode = int(returncode_match.group(1))
                # Remove the marker from output (handle various newline combinations)
                output_str = re.sub(r'\n*__RETURNCODE__=\d+\n*', '', output_str).strip()
            elif exec_result.status == ExecutionStatus.COMPLETED:
                # If no marker found but execution completed, default to 0
                returncode = 0
            
            if self.config.verbose:
                log_entry += f"\nRESULT:\n"
                log_entry += f"Status: {exec_result.status.value}\n"
                log_entry += f"Return Code: {returncode}\n"
                log_entry += f"Output:\n{output_str}\n"
                log_entry += f"{'='*60}\n"
                
                print(f"✅ RESULT (Step {self.step_count}):")
                print(f"Status: {exec_result.status.value}")
                print(f"Output preview: {output_str[:200]}")
                if len(output_str) > 200:
                    print(f"... (truncated, total length: {len(output_str)} chars)")
                print(f"{'='*60}\n")
                
                # Write to log file
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_entry)
            
            # Prepare result in agent-expected format
            result = {
                "output": output_str,
                "returncode": returncode
            }
            
            # Check if task is finished (raises Submitted exception if complete)
            self._check_finished(result)
            
            # Convert result to agent-expected format
            return result
            
        except Submitted:
            # Re-raise Submitted exception to signal task completion
            raise
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            
            if self.config.verbose:
                log_entry += f"\nERROR:\n{error_msg}\n"
                log_entry += f"{'='*60}\n"
                
                print(f"❌ ERROR (Step {self.step_count}):")
                print(error_msg)
                print(f"{'='*60}\n")
                
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_entry)
            
            return {
                "output": error_msg,
                "returncode": 1
            }
    
    def _handle_notebook_operation(self, command: str) -> dict[str, Any]:
        """
        Handle special notebook operations that need to run outside the container.
        
        Format: __NOTEBOOK_OP__<operation>(<args>)
        
        Supported operations:
        - get_cell_count(): Returns number of cells
        - get_cells(): Returns all cells
        - get_cell(index): Returns specific cell
        - edit_cell(index, code): Edits a cell
        - run_cell(index): Runs a specific cell
        - run_all(): Runs all cells
        - save(): Saves the notebook
        """
        try:
            # Parse the operation
            command = command.strip().replace("__NOTEBOOK_OP__", "")
            
            result = None
            
            # Handle different operations
            if command == "get_cell_count()":
                count = self.notebook.get_cell_count()
                result = {"output": str(count), "returncode": 0}
            
            elif command == "get_cells()":
                cells = self.notebook.get_cells()
                output = "\n\n".join(cells)
                result = {"output": output, "returncode": 0}
            
            elif command.startswith("get_cell("):
                index = int(command.split("(")[1].split(")")[0])
                cell = self.notebook.get_cell(index)
                result = {"output": cell, "returncode": 0}
            
            elif command.startswith("edit_cell("):
                # Parse: edit_cell(index, "code")
                import ast
                # Extract args safely
                args_str = command[len("edit_cell("):-1]
                parts = args_str.split(",", 1)
                index = int(parts[0].strip())
                code = ast.literal_eval(parts[1].strip())
                self.notebook.edit_cell(index, code)
                result = {"output": f"Cell {index} edited successfully", "returncode": 0}
            
            elif command.startswith("run_cell("):
                index = int(command.split("(")[1].split(")")[0])
                
                # Run asynchronously
                if platform.system() == 'Windows':
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                else:
                    loop = asyncio.get_event_loop()
                
                exec_result = loop.run_until_complete(
                    self.notebook.run_cell_async(index, cancel_event=None)
                )
                
                # Extract output as clean string from ExecutionResult
                if exec_result.result:
                    parts = exec_result.result.llm_compatible()
                    cell_output = _result_parts_to_string(parts)
                else:
                    cell_output = "(No output)"
                
                # Get status label
                status = _get_status_label(exec_result)
                output_text = f"Cell {index} [{status}]:\n{cell_output}"
                
                result = {
                    "output": output_text,
                    "returncode": 0 if exec_result.status == ExecutionStatus.COMPLETED else 1
                }
            
            elif command == "run_all()":
                # Run all cells
                if platform.system() == 'Windows':
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                else:
                    loop = asyncio.get_event_loop()
                
                exec_results = loop.run_until_complete(
                    self.notebook.run_all_async(cancel_event=None)
                )
                
                # Extract output as clean string from each ExecutionResult
                output_parts = []
                for i, exec_result in enumerate(exec_results):
                    if exec_result.result:
                        parts = exec_result.result.llm_compatible()
                        cell_output = _result_parts_to_string(parts)
                    else:
                        cell_output = "(No output)"
                    
                    # Get status label using shared function
                    status = _get_status_label(exec_result)
                    output_parts.append(f"Cell {i} [{status}]:\n{cell_output}")
                
                output = "\n\n".join(output_parts)
                all_success = all(r.status == ExecutionStatus.COMPLETED for r in exec_results)
                result = {
                    "output": output,
                    "returncode": 0 if all_success else 1
                }
            
            elif command == "save()":
                self.notebook.save()
                result = {"output": "Notebook saved successfully", "returncode": 0}
            
            else:
                result = {
                    "output": f"Unknown notebook operation: {command}",
                    "returncode": 1
                }
            
            # Log notebook operation result
            if result and self.config.verbose:
                print(f"{'='*60}\n")
                print(f"📓 NOTEBOOK OPERATION (Step {self.step_count}):")
                print(f"Operation: {command[:100]}")
                print(f"Result: {result['output'][:200]}")
                if len(result['output']) > 200:
                    print(f"... (truncated)")
                print(f"{'='*60}\n")
                
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{'='*60}\n")
                        f.write(f"\n📓 NOTEBOOK OPERATION:\n")
                        f.write(f"Operation: {command}\n")
                        f.write(f"Result:\n{result['output']}\n")
                        f.write(f"{'='*60}\n")
            
            return result
                
        except Exception as e:
            return {
                "output": f"Notebook operation error: {str(e)}",
                "returncode": 1
            }
    
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
        if self.notebook:
            template_vars.update({
                "notebook_initialized": True,
                "cell_count": self.notebook.get_cell_count(),
                
                # Instructions for getting notebook content
                "notebook_access_hint": (
                    "Use __NOTEBOOK_OP__get_cells() to view all notebook cells. "
                    f"The notebook has {self.notebook.get_cell_count()} cells."
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
            raise Submitted(submission)
    
    def cleanup(self):
        """Cleanup the Docker container and resources."""
        if self.problem:
            self.problem.teardown()
            self.problem = None
            self.notebook = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()

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
# Print stdout
print(result.stdout, end='')
# Print stderr to stderr stream
if result.stderr:
    import sys
    print(result.stderr, end='', file=sys.stderr)
# Print returncode marker on its own line for parsing
print(f'\\n__RETURNCODE__={{result.returncode}}')
"""