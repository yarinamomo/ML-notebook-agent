"""
Summary logger for agent execution tracking.
Creates a human-readable summary of the agent's execution.
"""
from pathlib import Path
from typing import Any
import json
from datetime import datetime

class SummaryLogger:
    """Tracks agent execution and generates summarized logs."""
    
    def __init__(self, output_path: Path | str | None = None):
        self.output_path = Path(output_path) if output_path else Path("./trajectories/last_run_summary.json")
        self.steps: list[dict] = []
        self.operations: list[dict] = []
        self.success = False
        self.start_time = datetime.now()
        self.original_cells: dict[int, str] = {}  # cell_id -> original source
        self.edited_cells: dict[int, list[str]] = {}    # cell_id -> edited source
        self.cost = 0.0  # Total cost for this instance
        
    def log_llm_response(self, content: str, reasoning: str, step: int) -> None:
        """Log an LLM response."""
        self.steps.append({
            "step": step,
            "content": content,
            "reasoning": reasoning
        })
    
    def log_operation(self, action: str, output: str, return_code: int, step: int) -> None:
        """Log an operation/action."""
        # Use the same step counter from the LLM response
        self.operations.append({
            "step": step,
            "action": action,
            "output": output,
            "return_code": return_code,
            "success": return_code == 0
        })
        self.success = action == "Submitted" and return_code == 0
    
    def log_cell_edit(self, cell_id: int, original: str, edited: str) -> None:
        """Log a cell edit."""
        if cell_id not in self.original_cells:
            self.original_cells[cell_id] = original
        self.edited_cells[cell_id] = self.edited_cells.get(cell_id, []) + [edited]
    
    def set_cost(self, cost: float) -> None:
        """Set the total cost for this instance."""
        self.cost = cost
    
    def save_summary(self) -> None:
        """Generate and save the summary as JSON."""
        summary = self._generate_summary()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    def _generate_summary(self) -> dict[str, Any]:
        """Generate the summary as a dictionary."""
        end_time = datetime.now()
        
        # Build code changes list
        code_changes = []
        for cell_id in sorted(self.edited_cells.keys()):
            original = self.original_cells.get(cell_id, "")
            edits = self.edited_cells[cell_id]
            code_changes.append({
                "cell_id": cell_id,
                "original_code": original,
                "fixed_code": edits[-1] if edits else original,
                "steps": [original] + edits,
            })
        
        # Count successful operations
        successful_ops = sum(1 for op in self.operations if op.get("success", False))
        
        return {
            "metadata": {
                "generated_at": end_time.isoformat(),
                "execution_time_seconds": (end_time - self.start_time).total_seconds(),
                "success": self.success,
                "status": "SUCCESS" if self.success else "INCOMPLETE",
                "cost": round(self.cost, 4)
            },
            "statistics": {
                "total_steps": len(self.steps),
                "total_operations": len(self.operations),
                "successful_operations": successful_ops,
                "failed_operations": len(self.operations) - successful_ops,
                "cells_edited": len(self.edited_cells)
            },
            "llm_responses": self.steps,
            "operations": self.operations,
            "code_changes": code_changes
        }


class NoOpSummary(SummaryLogger):
    """No-op logger used when logging is disabled. All methods do nothing."""

    def __init__(self) -> None:
        # Minimal init — no state needed since nothing is ever recorded
        pass

    def log_llm_response(self, content: str, reasoning: str, step: int) -> None:
        pass

    def log_operation(self, action: str, output: str, return_code: int, step: int) -> None:
        pass

    def log_cell_edit(self, cell_id: int, original: str, edited: str) -> None:
        pass

    def mark_success(self, step: int) -> None:
        pass

    def set_cost(self, cost: float) -> None:
        pass

    def save_summary(self) -> None:
        pass


# Global instance
_logger_instance: SummaryLogger | None = None


def get_summary() -> SummaryLogger:
    """Get the global summary logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = NoOpSummary()
    return _logger_instance


def initialize_logger(enabled: bool = False, output_path: Path | str | None = None) -> SummaryLogger:
    """Initialize the global summary logger."""
    global _logger_instance
    if enabled:
        _logger_instance = SummaryLogger(output_path=output_path)
    else:
        _logger_instance = NoOpSummary()
    return _logger_instance
