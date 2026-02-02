import asyncio
from dataclasses import dataclass
from typing import List, Optional
from .sandbox import DockerSandbox, ExecutionStatus, SandboxResult
from pathlib import Path
import shutil
import re
import os
import time

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
    def __init__(self, content: str, sandbox: DockerSandbox, problem_source_path: Path, problem_file: Path, with_debugger: bool = False):
        self.sandbox = sandbox
        self.cells = self._parse_cells(content)
        self.state = [None for _ in range(len(self.cells))]
        self.problem_source_path = problem_source_path
        self.problem_file = problem_file
        if with_debugger:
            self.add_debugger_to_problem()

    def add_debugger_to_problem(self) -> str:
        if  self.cells and len(self.cells) > 1:
            self.cells[0] = "import pymcdebug as pmd\n" + self.cells[0]
            
            for cell_index in range(0, len(self.cells)):
                self.cells[cell_index] = self.cells[cell_index].replace("pm.sample(", "pmd.debug(")

        self.save()

    def _parse_cells(self, content: str) -> List[str]:
        # Split by the cell marker #%%
        # The first element might be empty or imports before the first cell
        raw_cells = re.split(r'\n?\s*#\s*%%', content)

        cells = []
        for cell in raw_cells:
            cell = cell.strip()
            if cell:
                # Remove the header if it exists
                cell = re.sub(r'^# --- \[CELL \d+\]: ---\s*', '', cell)
                cells.append(cell)
        return cells

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
            result = self.sandbox.run(code)
            self.state[index] = result
            return result
        raise IndexError("Cell index out of range")

    def run_all(self) -> List[SandboxResult]:
        results = []
        for i in range(len(self.cells)):
            results.append(self.run_cell(i))
        return results
    
    async def run_cell_async(self, index: int, cancel_event: Optional[asyncio.Event] = None):
        if 0 <= index < len(self.cells):
            # Change working directory inside the Docker container before executing code
            code = f"import os\nos.chdir('/app/container')\n{self.cells[index]}"
            result = await self.sandbox.run_async(code, cancel_event=cancel_event)
            self.state[index] = result
            return result
        raise IndexError("Cell index out of range")
    
    async def run_all_async(self, cancel_event: Optional[asyncio.Event] = None) -> List[SandboxResult]:
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
        self.problem_file.write_text(self.to_script())

class BenchmarkProblem:
    def __init__(self, sandbox_settings: dict, source_path: str, docker_source_path: str = "docker_source"):
        self.source_path = Path(source_path)
        self.docker_source_path = Path(docker_source_path).resolve()
        self.sandbox = None
        self.notebook = None
        self.sandbox_settings = sandbox_settings

    def setup(self, with_debugger: bool = False):
        setup_environment(self.source_path, self.docker_source_path)

        self.sandbox = DockerSandbox(mount_volume=str(self.docker_source_path), **self.sandbox_settings)
        self.sandbox.start()

        problem_file = self.docker_source_path / "problem.py"
        if not problem_file.exists():
            raise FileNotFoundError(f"problem.py not found in {self.docker_source_path}")
        
        content = problem_file.read_text()

        self.sandbox.run("import sys\nsys.modules['__main__'].__file__ = '/app/container/problem.py'")

        self.notebook = LightweightNotebook(content, self.sandbox, self.docker_source_path, problem_file, with_debugger=with_debugger)

    def teardown(self):
        if self.sandbox:
            self.sandbox.stop()
            self.sandbox = None