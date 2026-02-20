"""
Summary logger for agent execution tracking.
Creates a human-readable summary of the agent's execution.
"""
from pathlib import Path
from typing import Any
import json
from datetime import datetime
from src.utils.log import logger as main_logger


class SummaryLogger:
    """Tracks agent execution and generates summarized logs."""
    
    def __init__(self, enabled: bool = False, output_path: Path | str | None = None):
        self.enabled = enabled
        self.output_path = Path(output_path) if output_path else Path("./trajectories/last_run_summary.json")
        self.steps = []
        self.operations = []
        self.success = False
        self.start_time = datetime.now()
        self.original_cells = {}  # cell_id -> original source
        self.edited_cells = {}    # cell_id -> edited source
        self.cost = 0.0  # Total cost for this instance
        
    def log_llm_response(self, content: str, step: int) -> None:
        """Log an LLM response."""
        if not self.enabled:
            return
        self.steps.append({
            "step": step,
            "content": content
        })
    
    def log_operation(self, action: str, output: str, return_code: int, step: int) -> None:
        """Log an operation/action."""
        if not self.enabled:
            return
        self.operations.append({
            "step": step,
            "action": action,
            "output": output,
            "return_code": return_code,
            "success": return_code == 0
        })
    
    def log_cell_edit(self, cell_id: int, original: str, edited: str) -> None:
        """Log a cell edit."""
        if not self.enabled:
            return
        if cell_id not in self.original_cells:
            self.original_cells[cell_id] = original
        self.edited_cells[cell_id] = edited
    
    def mark_success(self, step: int) -> None:
        """Mark the execution as successfully completed."""
        if not self.enabled:
            return
        self.success = True
        self.operations.append({
            "step": step,
            "action": "Job Submitted",
            "output": "Task completed successfully",
            "return_code": 0,
            "success": True
        })
    
    def set_cost(self, cost: float) -> None:
        """Set the total cost for this instance."""
        if not self.enabled:
            return
        self.cost = cost
    
    def save_summary(self) -> None:
        """Generate and save the summary as JSON."""
        if not self.enabled:
            return
        
        summary = self._generate_summary()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    def _generate_summary(self) -> dict[str, Any]:
        """Generate the summary as a dictionary."""
        end_time = datetime.now()
        
        # Build code changes list
        code_changes = []
        for cell_id in sorted(self.edited_cells.keys()):
            code_changes.append({
                "cell_id": cell_id,
                "original_code": self.original_cells.get(cell_id, ""),
                "fixed_code": self.edited_cells[cell_id]
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
            "llm_responses": [
                {
                    "step": step_data["step"],
                    "content": step_data["content"],
                }
                for step_data in self.steps
            ],
            "operations": [
                {
                    "step": op["step"],
                    "action": op["action"],
                    "output": op["output"],
                    "return_code": op["return_code"],
                    "success": op.get("success", False)
                }
                for op in self.operations
            ],
            "code_changes": code_changes
        }


# Global instance
_logger_instance: SummaryLogger | None = None


def get_logger() -> SummaryLogger:
    """Get the global summary logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SummaryLogger(enabled=False)
    return _logger_instance


def initialize_logger(enabled: bool = False, output_path: Path | str | None = None) -> SummaryLogger:
    """Initialize the global summary logger."""
    global _logger_instance
    _logger_instance = SummaryLogger(enabled=enabled, output_path=output_path)
    return _logger_instance


def print_execution_summary(results_summary: list[dict[str, Any]]) -> None:
    """
    Print execution summary to logger.
    
    Args:
        results_summary: List of result dictionaries from all runs
    """
    main_logger.info(f"\n{'='*80}")
    main_logger.info("EXECUTION SUMMARY")
    main_logger.info(f"{'='*80}")
    main_logger.info(f"Total runs: {len(results_summary)}")
    
    successful = sum(1 for r in results_summary if r.get('success', False))
    failed = sum(1 for r in results_summary if not r.get('success', False))
    
    main_logger.info(f"Successful: {successful}")
    main_logger.info(f"Failed: {failed}")
    
    # Break down by model if multiple models
    models = set(r.get('model', 'unknown') for r in results_summary)
    if len(models) > 1:
        main_logger.info(f"\nBreakdown by model:")
        for model in sorted(models):
            model_results = [r for r in results_summary if r.get('model') == model]
            model_success = sum(1 for r in model_results if r.get('success', False))
            main_logger.info(f"  {model}: {model_success}/{len(model_results)} successful")


def save_overall_summary(results_summary: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save overall summary of all runs to JSON file.
    
    Args:
        results_summary: List of result dictionaries from all runs
        output_path: Path to save the summary
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    main_logger.info(f"Overall summary saved to: {output_path}")
