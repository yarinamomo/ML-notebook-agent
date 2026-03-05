from typing import Any, List, Optional
from pathlib import Path
import shutil
import time
import src.utils.preprocess_notebook as preprocess_notebook
import src.utils.nbformat_helper as nbformat_helper
from src.utils.nb_types import CellExecutionResult
from src.sandbox import DockerSandbox
import logging


def setup_environment(src, dst):
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src.resolve()}")

    # Make sure destination is fresh (remove if exists)
    if dst.exists():
        if dst.is_file():
            dst.unlink()
        else:
            _remove_directory_with_retry(dst, max_retries=3, delay=1.0)

    # Copy recursively
    shutil.copytree(src, dst)
    
    # Delay to ensure filesystem sync for Docker volume mounts on macOS
    # Docker Desktop can have delays seeing newly created files after rmtree + copytree
    time.sleep(5.0) # 5x longer than tests needed to be absolutely certain

def _remove_directory_with_retry(path: Path, max_retries: int = 3, delay: float = 1.0):
    """Remove directory with retry logic for Docker mounted volumes.
    
    Args:
        path: Directory path to remove
        max_retries: Number of retry attempts
        delay: Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            # Use ignore_errors=True for problematic files (like .DS_Store on macOS)
            shutil.rmtree(path, ignore_errors=True)
            # Verify directory is actually gone
            if not path.exists():
                return
        except OSError as e:
            if attempt < max_retries - 1:
                logging.warning(f"Failed to remove {path} (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                time.sleep(delay)
            else:
                raise
    
    # Final check - if directory still exists, raise error
    if path.exists():
        raise OSError(f"Failed to remove directory {path} after {max_retries} attempts")

# actual environment for the agent
class BenchmarkProblem:
    def __init__(self, sandbox_settings: dict, source_path: Path, output_dir: Path, problem_mode: str = "JunoBench_Buggy", docker_source_path: str = "docker_source", timeout: int = 600):
        self.source_path = source_path
        self.docker_source_path = Path(docker_source_path).resolve() / f"sandbox_{sandbox_settings['port']}"
        self.output_dir = output_dir
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

    def run_cell(self, index: int) -> CellExecutionResult:
        code = self._get_cell_source(index)
        # Change working directory inside the Docker container before executing code
        code = f"import os\nos.chdir('/app/container')\n{code}" # TODO this should be moved into sandbox.
        result = self.sandbox.run(code, timeout=self.timeout)
        self._exec_states[index] = result
        nbformat_helper.save_cells(self._cells, self._original_cells, self._cell_states, self._exec_states, self.source_path.name, self.output_dir)
        return result

    def run_all(self) -> List[CellExecutionResult]:
        self.sandbox.restart_kernel()
        results = []
        for i in range(len(self._cells)):
            result: CellExecutionResult = self.run_cell(i)
            results.append(result)
            if result["status"] in ["error", "timeout"]:
                # If a cell fails, we stop execution and return results so far
                logging.info(f"Cell {i} execution failed. Stopping run_all." + (f" Last output: {result.get('outputs', [])[-1]}" if result.get('outputs', []) else ""))
                break
        return results


    def execute_python_command(self, command: str):
        if not self._cells:
            raise RuntimeError("Notebook not initialized")
        exec_result = self.sandbox.run(command)
        # print(f"Raw execution result: {exec_result}")
        return exec_result

    def close(self):
        # Stop sandbox first to release Docker volume mounts
        if self.sandbox:
            try:
                self.sandbox.stop()
            except Exception as e:
                logging.warning(f"Error stopping sandbox: {e}")
        
        # Give Docker time to release volume locks
        time.sleep(0.5)
        
        # Clean up the mount path (docker_source_path)
        if self.docker_source_path and self.docker_source_path.exists():
            try:
                _remove_directory_with_retry(self.docker_source_path, max_retries=3, delay=1.0)
            except Exception as e:
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
