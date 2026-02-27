from typing import Any, List, Optional
from pathlib import Path
import shutil
import time
import threading
import src.utils.preprocess_notebook as preprocess_notebook
import src.utils.nbformat_helper as nbformat_helper
from src.utils.nb_types import CellExecutionResult
from src.sandbox import DockerSandbox

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

# actual environment for the agent
class BenchmarkProblem:
    def __init__(self, sandbox_settings: dict, source_path: str, output_dir: str, problem_mode: str = "JunoBench_Buggy", docker_source_path: str = "docker_source", timeout: int = 30, run_all_timeout: int = 0):
        self.source_path = Path(source_path)
        self.docker_source_path = Path(docker_source_path).resolve()
        self.output_dir = Path(output_dir)
        setup_environment(self.source_path, self.docker_source_path)
        # Extract target_nb_instance from source_path
        target_nb_instance = self.source_path.name
        self.problem_file = self.docker_source_path / f"{target_nb_instance}_reproduced.ipynb"
        if not self.problem_file.exists():
            raise FileNotFoundError(f"{target_nb_instance}_reproduced.ipynb not found in {self.docker_source_path}")

        self.sandbox: DockerSandbox = DockerSandbox(mount_volume=str(self.docker_source_path), **sandbox_settings)
        self.sandbox.start()  
        # self.sandbox.run("print(\"Hello World\")")
        self.sandbox.run(f"import sys\nsys.modules['__main__'].__file__ = '/app/container/{target_nb_instance}_reproduced.ipynb'")

        self._cells = nbformat_helper.select_code_cells(nbformat_helper.load_notebook(self.problem_file), problem_mode)
        # for cell_info in self._cells:
        #     print(cell_info)
        self._cell_states: dict[int, str] = dict()  # cell_id -> state (edited/unchanged)
        self._exec_states: dict[int, CellExecutionResult] = dict()  # cell_id -> execution result
        # Store original cell contents for tracking true original state
        self._original_cells: dict[int, str] = {i: self._get_cell_source(i) for i in range(len(self._cells))}
        self.timeout = timeout # TODO Move into sandbox settings.
        self.run_all_timeout = run_all_timeout  # Total timeout for run_all() operation 

    def get_cells(self):
        return [self.get_cell(i) for i in range(len(self._cells))]

    def get_cell_count(self):
        return len(self._cells)

    def get_cell(self, index: int):
        return f"# --- [CELL {index}]: ---\n{self._get_cleaned_cell_source(index)}"

    def edit_cell(self, index: int, new_content: str):
        cell = self._safe_get_cell(index)
        original_content = self._original_cells.get(index, "")
        
        cell["source"] = new_content
        self._cell_states[index] = "edited"
        self._exec_states.pop(index, None) # reset execution state since content changed
        nbformat_helper.save_cells(self._cells, self._original_cells, self._cell_states, self._exec_states, self.source_path.name, self.output_dir)

    def run_cell(self, index: int):
        code = self._get_cell_source(index)
        # Change working directory inside the Docker container before executing code
        code = f"import os\nos.chdir('/app/container')\n{code}" # TODO this should be moved into sandbox.
        result = self.sandbox.run(code, timeout=self.timeout)
        self._exec_states[index] = result
        nbformat_helper.save_cells(self._cells, self._original_cells, self._cell_states, self._exec_states, self.source_path.name, self.output_dir)
        return result

    def run_all(self) -> List[CellExecutionResult]:
        self.sandbox.restart_kernel()
        
        # If run_all_timeout is set, run with timeout
        if self.run_all_timeout and self.run_all_timeout > 0:
            results = []
            timeout_occurred = False
            exception = None
            
            def run_cells():
                nonlocal results, exception
                try:
                    for i in range(len(self._cells)):
                        results.append(self.run_cell(i))
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_cells)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.run_all_timeout)
            
            if thread.is_alive():
                # Timeout occurred
                timeout_occurred = True
                # Create a timeout error result for remaining cells in CellExecutionResult format
                for i in range(len(results), len(self._cells)):
                    results.append({
                        "execution_count": i + 1,
                        "status": "error",
                        "outputs": [{
                            "output_type": "error",
                            "ename": "TimeoutError",
                            "evalue": f"run_all() operation timed out after {self.run_all_timeout} seconds. Cell {i} was not executed.",
                        }]
                    })
            
            if exception:
                raise exception
                
            nbformat_helper.save_cells(self._cells, self._original_cells, self._cell_states, self._exec_states, self.source_path.name, self.output_dir)
            return results
        else:
            # Run without timeout (original behavior)
            results = []
            for i in range(len(self._cells)):
                results.append(self.run_cell(i))
            nbformat_helper.save_cells(self._cells, self._original_cells, self._cell_states, self._exec_states, self.source_path.name, self.output_dir)
            return results


    def execute_python_command(self, command: str):
        if not self._cells:
            raise RuntimeError("Notebook not initialized")
        exec_result = self.sandbox.run(command, timeout=self.timeout)
        # print(f"Raw execution result: {exec_result}")
        return exec_result

    def close(self):
        if self.sandbox:
            self.sandbox.stop()
        
        # Clean up the mount path (docker_source_path)
        if self.docker_source_path and self.docker_source_path.exists():
            try:
                shutil.rmtree(self.docker_source_path)
            except Exception as e:
                import logging
                logging.warning(f"⚠️ Warning: Could not clean up mount path {self.docker_source_path}: {e}")

    def _safe_get_cell(self, index: int):
        if 0 <= index < len(self._cells):
            return self._cells[index]
        raise IndexError("Cell index out of range")

    def _get_cell_source(self, index: int) -> str:
        cell = self._safe_get_cell(index)
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        return source
    
    def _get_cleaned_cell_source(self, index: int) -> str:
        return preprocess_notebook.remove_comments(self._get_cell_source(index))
