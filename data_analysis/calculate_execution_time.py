import json
from pathlib import Path

include_settings = [
    "results_JunoBench/agent",
    "results_JunoBench/agent_without_run_code_and_cell_outputs",
    "results_JunoBench/baseline_without_cell_outputs",
    "results_JunoBench/baseline_with_all_outputs",
    "results_JunoBench/baseline_without_all_outputs",
    "results_new/agent",
    "results_new/agent_without_run_code_and_cell_outputs",
    "results_new/baseline_without_cell_outputs",
]
include_models = ["gpt-5.3-codex", "glm-4.7-355b"]
    
def calculate_execution_times():
    execution_times = {
        'junobench/glm': 0,
        'junobench/codex': 0,
        'new/glm': 0,
        'new/codex': 0,
    }
    
    for setting in include_settings:
        for model in include_models:
            path = Path(setting) / model / "overall_summary.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    if "summary" in data and "total_execution_time_seconds" in data["summary"]:
                        time_val = data["summary"]["total_execution_time_seconds"]
                        # Categorize by junobench/new and glm/codex
                        bench_type = "junobench" if "JunoBench" in setting else "new"
                        model_type = "glm" if "glm" in model else "codex"
                        key = f"{bench_type}/{model_type}"
                        execution_times[key] += time_val
                        # print(f"{setting}/{model}: {time_val} seconds")
    
    print("\nSummary by category:")
    total = 0
    for key in ['junobench/glm', 'junobench/codex', 'new/glm', 'new/codex']:
        seconds = execution_times[key]
        minutes = seconds / 60
        hours = seconds / 3600
        print(f"  {key}: {seconds:.2f}s ({minutes:.2f}m / {hours:.2f}h)")
        total += seconds
    
    total_minutes = total / 60
    total_hours = total / 3600
    print(f"  Total: {total:.2f}s ({total_minutes:.2f}m / {total_hours:.2f}h)")


def calculate_total_runs():
    total_runs = {
        'junobench/glm': 0,
        'junobench/codex': 0,
        'new/glm': 0,
        'new/codex': 0,
    }
    
    for setting in include_settings:
        for model in include_models:
            path = Path(setting) / model / "overall_summary.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    if "summary" in data and "total_runs" in data["summary"]:
                        runs_val = data["summary"]["total_runs"]
                        # Categorize by junobench/new and glm/codex
                        bench_type = "junobench" if "JunoBench" in setting else "new"
                        model_type = "glm" if "glm" in model else "codex"
                        key = f"{bench_type}/{model_type}"
                        total_runs[key] += runs_val
    
    print("\n\nTotal Runs Summary by category:")
    grand_total = 0
    for key in ['junobench/glm', 'junobench/codex', 'new/glm', 'new/codex']:
        runs = total_runs[key]
        print(f"  {key}: {runs}")
        grand_total += runs
    print(f"  Total: {grand_total}")