import json
from pathlib import Path

def collect_results(run_dir):
    results = []
    # Find all subfolders named run_* and sort by run number
    run_folders = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                         key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0)
    for run_path in run_folders:
        try:
            run_num = int(run_path.name.split("_")[-1])
        except Exception:
            run_num = 0
        for file in run_path.glob("*.json"):
            if file.name.endswith("_summary.json"):
                with open(file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        # If this is a summary file, try to extract the main result
                        if isinstance(data, dict) and "metadata" in data and "statistics" in data:
                            # Try to extract relevant fields for overall summary
                            status = data["metadata"].get("status", "UNKNOWN")
                            total_steps = data["statistics"].get("total_steps", 0)
                            execution_time_seconds = data["metadata"].get("execution_time_seconds", 0.0)
                            
                            # Parse INCOMPLETE status more specifically
                            if status == "INCOMPLETE":
                                if total_steps >= 30:
                                    status = "LimitsExceeded"
                                elif execution_time_seconds >= 3600:
                                    status = "AgentTimeout"
                                else:
                                    status = "EnvironmentUnavailable"
                            
                            result = {
                                "model": run_dir.name,
                                "instance": file.stem.replace("_summary", ""),
                                "run": run_num,
                                "exit_status": status,
                                "success": data["metadata"].get("success", False),
                                "cost": data["metadata"].get("cost", 0.0),
                                "total_steps": total_steps,
                                "execution_time_seconds": execution_time_seconds,
                                "operations": data.get("operations", [])
                            }
                            results.append(result)
                    except Exception as e:
                        print(f"Failed to load {file}: {e}")
    return results

def generate_overall_summary(results, output_path):
    # Helper function to check if status is a submit status
    def is_submit_status(status):
        status_upper = status.upper()
        return status_upper in ['SUCCESS', 'SUBMITTED', 'SUBMITTEDWITHERRORS']
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = sum(1 for r in results if not r.get('success', False))
    total_cost = sum(r.get('cost', 0.0) for r in results)
    total_steps = sum(r.get('total_steps', 0) for r in results)
    total_execution_time_seconds = sum(r.get('execution_time_seconds', 0.0) for r in results)
    total_op_run_code = sum(
        sum(1 for op in r.get('operations', []) if "run_code" in op.get('action', ""))
        for r in results
    )
    # Determine number of runs (run_* folders) from results
    run_numbers = set(r.get('run', 0) for r in results)
    num_runs = len(run_numbers) if run_numbers else 1
    
    # Calculate pass@3: for each instance, check if it has a submit status in at least one run
    instance_results = {}
    for r in results:
        instance_name = r.get('instance', '')
        if instance_name not in instance_results:
            instance_results[instance_name] = []
        instance_results[instance_name].append(is_submit_status(r.get('exit_status', '')))
    
    # Count unique instances (instances that exist in at least one run folder)
    number_of_instances = len(instance_results)
    
    # Count instances that passed at least once across all runs (pass@k)
    pass_at_k_count = sum(1 for instance, successes in instance_results.items() if any(successes))
    pass_at_k_rate = pass_at_k_count / number_of_instances if number_of_instances else 0.0
    
    # Count instances that passed in ALL runs (pass_all_k)
    pass_all_k_count = sum(1 for instance, successes in instance_results.items() if all(successes))
    pass_all_k_rate = pass_all_k_count / number_of_instances if number_of_instances else 0.0
    
    # Calculate pass@k and pass_all_k per library
    library_instance_results = {}
    for instance, successes in instance_results.items():
        # Extract library name (e.g., "torch_1" -> "torch", "sklearn_10" -> "sklearn")
        library = instance.rsplit('_', 1)[0] if '_' in instance else instance
        if library not in library_instance_results:
            library_instance_results[library] = []
        library_instance_results[library].append(successes)
    
    pass_at_k_per_library = {}
    for library, successes_list in library_instance_results.items():
        lib_total = len(successes_list)
        lib_passed_any = sum(1 for s in successes_list if any(s))
        lib_passed_all = sum(1 for s in successes_list if all(s))
        pass_at_k_per_library[library] = {
            "pass_at_k_count": lib_passed_any,
            "pass_at_k_rate": lib_passed_any / lib_total if lib_total else 0.0,
            "pass_all_k_count": lib_passed_all,
            "pass_all_k_rate": lib_passed_all / lib_total if lib_total else 0.0,
            "total_instances": lib_total
        }
    # Helper functions
    def median(lst):
        lst = sorted(lst)
        n = len(lst)
        if n == 0:
            return 0
        if n % 2 == 1:
            return lst[n // 2]
        else:
            return (lst[n // 2 - 1] + lst[n // 2]) / 2

    costs = [r.get('cost', 0.0) for r in results]
    exec_times = [r.get('execution_time_seconds', 0.0) for r in results]
    steps = [r.get('total_steps', 0) for r in results]
    failed_instances = [f"run_{r['run']}/{r['instance']}" for r in results if not r.get("success", False)]
    success_instances = [f"run_{r['run']}/{r['instance']}" for r in results if r.get("success", False)]
    
    # Per-run statistics
    per_run_stats = {}
    for run_num in sorted(run_numbers):
        run_results = [r for r in results if r.get('run') == run_num]
        if not run_results:
            continue
        
        run_costs = [r.get('cost', 0.0) for r in run_results]
        run_exec_times = [r.get('execution_time_seconds', 0.0) for r in run_results]
        run_steps = [r.get('total_steps', 0) for r in run_results]
        run_successful = sum(1 for r in run_results if r.get('success', False))
        run_failed = len(run_results) - run_successful
        
        # Status distribution for this run
        run_incomplete_statuses = {}
        run_submit_statuses = {}
        for r in run_results:
            status = r.get('exit_status', 'UNKNOWN')
            if is_submit_status(status):
                run_submit_statuses[status] = run_submit_statuses.get(status, 0) + 1
            else:
                run_incomplete_statuses[status] = run_incomplete_statuses.get(status, 0) + 1
        
        # Run code instances for this run
        run_op_run_code_instances = [
            r['instance'] for r in run_results
            if any("run_code" in op.get("action", "") for op in r.get("operations", []))
        ]
        run_op_run_code_count = sum(
            sum(1 for op in r.get('operations', []) if "run_code" in op.get('action', ""))
            for r in run_results
        )
        
        per_run_stats[f"run_{run_num}"] = {
            "success_rate": run_successful / len(run_results) if run_results else 0.0,
            "status_distribution": {
                "incomplete_statuses": run_incomplete_statuses,
                "submit_statuses": run_submit_statuses,
            },
            "cost_distribution": {
                "total_cost": round(sum(run_costs), 4),
                "avg_cost": sum(run_costs) / len(run_results) if run_results else 0.0,
                "max_cost": max(run_costs) if run_costs else 0.0,
                "min_cost": min(run_costs) if run_costs else 0.0,
                "median_cost": median(run_costs),
            },
            "execution_time_distribution": {
                "total_execution_time_seconds": sum(run_exec_times),
                "avg_execution_time_seconds": sum(run_exec_times) / len(run_results) if run_results else 0.0,
                "max_execution_time_seconds": max(run_exec_times) if run_exec_times else 0.0,
                "min_execution_time_seconds": min(run_exec_times) if run_exec_times else 0.0,
                "median_execution_time_seconds": median(run_exec_times),
            },
            "steps_distribution": {
                "total_steps": sum(run_steps),
                "avg_steps": sum(run_steps) / len(run_results) if run_results else 0,
                "max_steps": max(run_steps) if run_steps else 0,
                "min_steps": min(run_steps) if run_steps else 0,
                "median_steps": median(run_steps),
            },
            "op_run_code_distribution": {
                "total_op_run_code": run_op_run_code_count,
                "avg_op_run_code_per_instance": run_op_run_code_count / len(run_results) if run_results else 0.0,
                "instances_with_run_code": run_op_run_code_instances,
            }
        }

    # Status distributions
    incomplete_statuses = {}
    submit_statuses = {}
    for r in results:
        status = r.get('exit_status', 'UNKNOWN')
        # Submit statuses include SUCCESS, Submitted, SubmittedWithErrors
        if is_submit_status(status):
            submit_statuses[status] = submit_statuses.get(status, 0) + 1
        else:
            # Everything else is incomplete
            incomplete_statuses[status] = incomplete_statuses.get(status, 0) + 1

    # Grouped statistics
    statistics = {
        "outcome_distribution": {
            "pass_at_k_rate": pass_at_k_rate,
            "pass_at_k_count": pass_at_k_count,
            "pass_all_k_rate": pass_all_k_rate,
            "pass_all_k_count": pass_all_k_count,
            "per_run": {run: {"success_rate": per_run_stats[run]["success_rate"]} for run in sorted(per_run_stats.keys())},
            "pass_at_k_per_library": pass_at_k_per_library,
            "failed_instances": failed_instances,
            "success_instances": success_instances,
        },
        "status_distribution": {
            "overall": {
                "incomplete_statuses": incomplete_statuses,
                "submit_statuses": submit_statuses,
            },
            "per_run": {run: per_run_stats[run]["status_distribution"] for run in sorted(per_run_stats.keys())}
        },
        "cost_distribution": {
            "overall": {
                "avg_cost": total_cost / len(results) if results else 0.0,
                "max_cost": max(costs) if costs else 0.0,
                "min_cost": min(costs) if costs else 0.0,
                "median_cost": median(costs),
                "avg_cost_per_success": sum(r.get('cost', 0.0) for r in results if r.get('success', False)) / successful if successful else 0.0,
                "avg_cost_per_failure": sum(r.get('cost', 0.0) for r in results if not r.get('success', False)) / failed if failed else 0.0,
            },
            "per_run": {run: per_run_stats[run]["cost_distribution"] for run in sorted(per_run_stats.keys())}
        },
        "execution_time_distribution": {
            "overall": {
                "avg_execution_time_seconds": total_execution_time_seconds / len(results) if results else 0.0,
                "max_execution_time_seconds": max(exec_times) if exec_times else 0.0,
                "min_execution_time_seconds": min(exec_times) if exec_times else 0.0,
                "median_execution_time_seconds": median(exec_times),
                "avg_execution_time_per_success": sum(r.get('execution_time_seconds', 0.0) for r in results if r.get('success', False)) / successful if successful else 0.0,
                "avg_execution_time_per_failure": sum(r.get('execution_time_seconds', 0.0) for r in results if not r.get('success', False)) / failed if failed else 0.0
            },
            "per_run": {run: per_run_stats[run]["execution_time_distribution"] for run in sorted(per_run_stats.keys())}
        },
        "steps_distribution": {
            "overall": {
                "avg_steps": total_steps / len(results) if results else 0,
                "max_steps": max(steps) if steps else 0,
                "min_steps": min(steps) if steps else 0,
                "median_steps": median(steps),
                "avg_steps_per_success": sum(r.get('total_steps', 0) for r in results if r.get('success', False)) / successful if successful else 0,
                "avg_steps_per_failure": sum(r.get('total_steps', 0) for r in results if not r.get('success', False)) / failed if failed else 0
            },
            "per_run": {run: per_run_stats[run]["steps_distribution"] for run in sorted(per_run_stats.keys())}
        },
        "op_run_code_distribution": {
            "overall": {
                "avg_op_run_code_per_instance": total_op_run_code / len(results) if results else 0.0,
            },
            "per_run": {run: per_run_stats[run]["op_run_code_distribution"] for run in sorted(per_run_stats.keys())}
        }
    }

    overall_summary = {
        "summary":{
            "number_of_instances": number_of_instances,
            "total_runs": len(results),
            "total_successful": successful,
            "total_successful_pass_at_k": pass_at_k_count,
            "total_successful_pass_all_k": pass_all_k_count,
            "total_failed": failed,
            "total_steps": total_steps,
            "total_cost": round(total_cost, 4),
            "total_execution_time_seconds": total_execution_time_seconds,
            "total_op_run_code": total_op_run_code
        },
        "statistics": statistics
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")

def main():
    # Configure base paths to process
    base_paths = [
        Path("trajectories_without_run_code/glm-4.7-355b")
    ]
    
    for base in base_paths:
        if not base.exists():
            print(f"Skipping {base} (does not exist)")
            continue
        
        # Collect results directly from the base path
        all_results = collect_results(base)
        
        if not all_results:
            print(f"No results found in {base}")
            continue
        
        output_path = base / "overall_summary.json"
        generate_overall_summary(all_results, output_path)

if __name__ == "__main__":
    main()
