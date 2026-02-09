import asyncio
from typing import List, Optional
from .sandbox import DockerSandbox, ExecutionStatus, SandboxResult, ExecutionResult
from pathlib import Path
import shutil
import re
import os
import time
import src.preprocess_notebook as preprocess_noteboook

def setup_environment(src, dst):
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src.resolve()}")

    # Make sure destination is fresh (remove if exists)
    if dst.exists():
        if dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    # Copy recursively
    shutil.copytree(src, dst)
    
    # Delay to ensure filesystem sync for Docker volume mounts on macOS
    # Docker Desktop can have delays seeing newly created files after rmtree + copytree
    time.sleep(5.0) # 5x longer than tests needed to be absolutely certain

class LightweightNotebook:
    def __init__(self, sandbox: DockerSandbox, problem_source_path: Path, problem_file: Path, problem_mode: str = "JunoBench_Buggy", timeout: int = 30):
        self.sandbox = sandbox
        self.cells = self._parse_cells(problem_file, problem_mode)
        self.state: List[Optional[ExecutionResult]] = [None for _ in range(len(self.cells))]
        self.problem_source_path = problem_source_path
        self.problem_file = problem_file
        self.timeout = timeout

    def _parse_cells(self, problem_file: Path, problem_mode: str) -> List[str]:
        res = preprocess_noteboook.parse_nb(problem_file, parse_mode=problem_mode)
        return res

    def get_cells(self):
        return [f"# --- [CELL {i}]: ---\n{cell}" for i, cell in enumerate(self.cells)]

    def get_cell_count(self):
        return len(self.cells)

    def get_cell(self, index: int):
        if 0 <= index < len(self.cells):
            return f"# --- [CELL {index}]: ---\n{self.cells[index]}"
        raise IndexError("Cell index out of range")

    def edit_cell(self, index: int, new_content: str):
        if 0 <= index < len(self.cells):
            self.cells[index] = new_content
            self.state[index] = None
            self.save()
        else:
            raise IndexError("Cell index out of range")

    # def add_cell(self, content: str, index: int = None):
    #     if index is None:
    #         self.cells.append(content)
    #         self.state.append(None)
    #     else:
    #         self.cells.insert(index, content)
    #         self.state.insert(index, None)

    #     self.save()

    # def remove_cell(self, index: int):
    #     if 0 <= index < len(self.cells):
    #         self.cells.pop(index)
    #         self.state.pop(index)
    #         self.save()
    #     else:
    #         raise IndexError("Cell index out of range")

    def run_cell(self, index: int):
        if 0 <= index < len(self.cells):
            # Change working directory inside the Docker container before executing code
            code = f"import os\nos.chdir('/app/container')\n{self.cells[index]}"
            result = self.sandbox.run(code, timeout=self.timeout)
            self.state[index] = result
            return result
        raise IndexError("Cell index out of range")

    def run_all(self) -> List[SandboxResult]:
        results = []
        for i in range(len(self.cells)):
            results.append(self.run_cell(i))
        return results
    
    async def run_cell_async(self, index: int, cancel_event: Optional[asyncio.Event] = None) -> ExecutionResult:
        if 0 <= index < len(self.cells):
            # Change working directory inside the Docker container before executing code
            code = f"import os\nos.chdir('/app/container')\n{self.cells[index]}"
            result = await self.sandbox.run_async(code, cancel_event=cancel_event, timeout=self.timeout)
            self.state[index] = result
            return result
        raise IndexError("Cell index out of range")
    
    async def run_all_async(self, cancel_event: Optional[asyncio.Event] = None) -> List[ExecutionResult]:
        results = []
        for i in range(len(self.cells)):
            result = await self.run_cell_async(i, cancel_event=cancel_event)
            results.append(result)

            if result.status == ExecutionStatus.CANCELLED:
                break
        return results
    
    def to_script(self):
        return "\n\n#%%\n".join([f"# --- [CELL {i}]: ---\n{cell}" for i, cell in enumerate(self.cells)])
    
    def save(self):
        # self.problem_file.write_text(self.to_script())
        pass

# actual environment for the agent
class BenchmarkProblem:
    def __init__(self, sandbox_settings: dict, source_path: str, problem_mode: str = "JunoBench_Buggy", docker_source_path: str = "docker_source", timeout: int = 30):
        self.source_path = Path(source_path)
        self.docker_source_path = Path(docker_source_path).resolve()
        self.sandbox = None
        self.notebook = None
        self.problem_mode = problem_mode
        self.sandbox_settings = sandbox_settings
        self.timeout = timeout

    def setup(self):
        setup_environment(self.source_path, self.docker_source_path)

        self.sandbox = DockerSandbox(mount_volume=str(self.docker_source_path), **self.sandbox_settings)
        self.sandbox.start()

        # Extract target_nb_instance from source_path
        target_nb_instance = self.source_path.name
        problem_file = self.docker_source_path / f"{target_nb_instance}_reproduced.ipynb"
        if not problem_file.exists():
            raise FileNotFoundError(f"{target_nb_instance}_reproduced.ipynb not found in {self.docker_source_path}")
        
        self.sandbox.run(f"import sys\nsys.modules['__main__'].__file__ = '/app/container/{target_nb_instance}_reproduced.ipynb'")

        self.notebook = LightweightNotebook(sandbox=self.sandbox, problem_source_path=self.docker_source_path, problem_file=problem_file, problem_mode=self.problem_mode, timeout=self.timeout)

    async def execute_python_command(self, command: str):
        if not self.notebook:
            result =  {
                "output": "Error: Notebook not initialized",
                "returncode": 1,
                "exception_info": "Notebook not initialized",
            }
        try:
            exec_result = await self.sandbox.run_async(command, cancel_event=None, timeout=self.timeout)
            # Extract output as clean string from ExecutionResult
            if exec_result.result:
                output_str = exec_result.result.llm_compatible()
            else:
                output_str = "(No output)"
            exception_info = ""
            if exec_result.status == ExecutionStatus.TIMEOUT:
                output_str = f"Timeout: Command timed out after {self.timeout} seconds."
                exception_info = f"Command timed out after {self.timeout} seconds."
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
            
            result = {
                "output": output_str,
                "returncode": returncode,
                "exception_info": exception_info
            }
            return result
        except Exception as e:
            return {
                "output": f"Error executing command: {str(e)}",
                "returncode": 1,
                "exception_info": f"Error executing command: {str(e)}",
            }

    async def execute_notebook_command(self, command: str):
        """
        Handle notebook operations initialized by the agent.
        
        Supported operations:
        - get_cell_count(): Returns number of cells
        - get_cells(): Returns all cells
        - get_cell(index): Returns specific cell
        - edit_cell(index, code): Edits a cell
        - run_cell(index): Runs a specific cell
        - run_all(): Runs all cells
        - save(): Saves the notebook
        """
        if not self.notebook:
            result =  {
                "output": "Error: Notebook not initialized",
                "returncode": 1,
                "exception_info": "Notebook not initialized",
            }
        
        try:
            if command == "get_cell_count()":
                count = self.notebook.get_cell_count()
                result = {
                    "output": str(count), 
                    "returncode": 0,
                    "exception_info": "",
                }
            
            elif command == "get_cells()":
                cells = self.notebook.get_cells()
                output = "\n\n".join(cells)
                result = {
                    "output": output, 
                    "returncode": 0,
                    "exception_info": "",
                }
            
            elif command.startswith("get_cell("):
                index = int(command.split("(")[1].split(")")[0])
                cell = self.notebook.get_cell(index)
                result = {
                    "output": cell, 
                    "returncode": 0, 
                    "exception_info": ""
                }
            
            elif command.startswith("edit_cell("):
                # Parse: edit_cell(index, "code")
                import ast
                # Extract args safely
                args_str = command[len("edit_cell("):-1]
                parts = args_str.split(",", 1)
                index = int(parts[0].strip())
                code = ast.literal_eval(parts[1].strip())
                self.notebook.edit_cell(index, code)
                result = {
                    "output": f"Cell {index} edited successfully", 
                    "returncode": 0, 
                    "exception_info": ""
                }
            
            elif command.startswith("run_cell("):
                index = int(command.split("(")[1].split(")")[0])
                
                # Run asynchronously
                exec_result = await self.notebook.run_cell_async(index, cancel_event=None)
                
                # Extract output as clean string from ExecutionResult
                if exec_result.result:
                    cell_output = exec_result.result.llm_compatible()
                else:
                    cell_output = "(No output)"
                
                exception_info = ""
                if exec_result.status == ExecutionStatus.TIMEOUT:
                    cell_output = f"Cell execution timed out after {self.timeout} seconds."
                    exception_info = f"Cell execution timed out after {self.timeout} seconds."

                # Get status label
                status = self._get_status_label(exec_result)
                output_text = f"Cell {index} [{status}]:\n{cell_output}"

                result = {
                    "output": output_text,
                    "returncode": 0 if exec_result.status == ExecutionStatus.COMPLETED else 1,
                    "exception_info": exception_info,
                }
            
            elif command == "run_all()":
                # Run all cells
                exec_results = await self.notebook.run_all_async(cancel_event=None)
                
                exception_info = ""
                # Extract output as clean string from each ExecutionResult
                output_parts = []
                for i, exec_result in enumerate(exec_results):
                    if exec_result.result:
                        cell_output = exec_result.result.llm_compatible(if_truncate=True, max_words=500)
                    else:
                        cell_output = "(No output)"
                    
                    if exec_result.status == ExecutionStatus.TIMEOUT:
                        cell_output = f"Cell execution timed out after {self.timeout} seconds."
                        exception_info = f"Cell execution timed out after {self.timeout} seconds."
                    # Get status label using shared function
                    status = self._get_status_label(exec_result)
                    output_parts.append(f"Cell {i} [{status}]:\n{cell_output}")
                
                output = "\n\n".join(output_parts)
                all_success = all(r.status == ExecutionStatus.COMPLETED for r in exec_results)
                result = {
                    "output": output,
                    "returncode": 0 if all_success else 1,
                    "exception_info": exception_info,
                }
            
            elif command == "save()":
                self.notebook.save()
                result = {
                    "output": "Notebook saved successfully", 
                    "returncode": 0,
                    "exception_info": "",
                    }
            
            else:
                result = {
                    "output": f"Unknown notebook operation: {command}",
                    "returncode": 1,
                    "exception_info": f"Unknown notebook operation: {command}",
                }
        except Exception as e:
            result = {
                "output": f"Error executing notebook command: {str(e)}",
                "returncode": 1,
                "exception_info": f"Error executing notebook command: {str(e)}",
            }
        return result

    def teardown(self):
        if self.sandbox:
            self.sandbox.stop()
            self.sandbox = None

    def _get_status_label(self, exec_result: ExecutionResult) -> str:
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
        elif exec_result.status == ExecutionStatus.TIMEOUT:
            return "TIMEOUT"
        else:
            return exec_result.status.value.upper()