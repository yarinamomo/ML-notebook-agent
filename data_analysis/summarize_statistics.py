import json
from pathlib import Path
from itertools import combinations
from math import comb, erf, sqrt
from data_analysis.analyze_patched_executed_failures import evaluate_notebook_correctness

def collect_results(run_dir):
    results = []
    # Find all subfolders named run_* and sort by run number
    run_folders = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                         key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0)
    for run_path in run_folders:
        try:
            run_num = int(run_path.name.split("_")[-1])
        except ValueError:
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
                            instance_name = file.stem.replace("_summary", "")
                            notebook_path = run_path / f"{instance_name}_patched_executed.ipynb"
                            is_correct, was_executed, error_name, error_preview = evaluate_notebook_correctness(notebook_path)
                            
                            result = {
                                "model": run_dir.name,
                                "instance": instance_name,
                                "run": run_num,
                                "exit_status": status,
                                "is_correct": is_correct,
                                "last_cell_was_executed": was_executed,
                                "last_cell_error_name": error_name,
                                "last_cell_error_preview": error_preview,
                                # "success": data["metadata"].get("success", False),
                                "cost": data["metadata"].get("cost", 0.0),
                                "total_steps": total_steps,
                                "execution_time_seconds": execution_time_seconds,
                                "operations": data.get("operations", [])
                            }
                            results.append(result)
                    except (OSError, json.JSONDecodeError, ValueError) as e:
                        print(f"Failed to load {file}: {e}")
    return results

def generate_overall_summary(results, output_path):
    # Helper function to check if status is a submit status
    def is_submit_status(status):
        status_upper = status.upper()
        return status_upper in ['SUCCESS', 'SUBMITTED', 'SUBMITTEDWITHERRORS']

    def is_plausible_result(result):
        return is_submit_status(result.get('exit_status', ''))

    def is_correct_result(result):
        return bool(result.get('is_correct', False))
    
    plausible = sum(1 for r in results if is_plausible_result(r))
    correct = sum(1 for r in results if is_correct_result(r))
    failed = len(results) - plausible
    incorrect = len(results) - correct
    total_cost = sum(r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in results)
    total_steps = sum(r.get('total_steps', 0) for r in results)
    total_execution_time_seconds = sum(r.get('execution_time_seconds', 0.0) for r in results)
    total_op_run_code = sum(
        sum(1 for op in r.get('operations', []) if "run_code" in op.get('action', ""))
        for r in results
    )
    # Determine number of runs (run_* folders) from results
    run_numbers = set(r.get('run', 0) for r in results)
    
    # Calculate pass@k style metrics across runs for plausible and correct outcomes.
    instance_results = {}
    instance_correct_results = {}
    for r in results:
        instance_name = r.get('instance', '')
        if instance_name not in instance_results:
            instance_results[instance_name] = []
        if instance_name not in instance_correct_results:
            instance_correct_results[instance_name] = []
        instance_results[instance_name].append(is_submit_status(r.get('exit_status', '')))
        instance_correct_results[instance_name].append(is_correct_result(r))
    
    # Count unique instances (instances that exist in at least one run folder)
    number_of_instances = len(instance_results)
    
    # Count instances that passed at least once across all runs (pass@k)
    plause_at_k_count = sum(1 for instance, plausible_cases in instance_results.items() if any(plausible_cases))
    plause_at_k_rate = plause_at_k_count / number_of_instances if number_of_instances else 0.0
    
    # Count instances that passed in ALL runs (pass_all_k)
    plause_all_k_count = sum(1 for instance, plausible_cases in instance_results.items() if all(plausible_cases))
    plause_all_k_rate = plause_all_k_count / number_of_instances if number_of_instances else 0.0

    correct_at_k_count = sum(1 for instance, correct_cases in instance_correct_results.items() if any(correct_cases))
    correct_at_k_rate = correct_at_k_count / number_of_instances if number_of_instances else 0.0

    correct_all_k_count = sum(1 for instance, correct_cases in instance_correct_results.items() if all(correct_cases))
    correct_all_k_rate = correct_all_k_count / number_of_instances if number_of_instances else 0.0
    
    # Calculate pass@k and pass_all_k per library
    library_instance_results = {}
    for instance, plausible_cases in instance_results.items():
        # Extract library name (e.g., "torch_1" -> "torch", "sklearn_10" -> "sklearn")
        library = instance.rsplit('_', 1)[0] if '_' in instance else instance
        if library not in library_instance_results:
            library_instance_results[library] = []
        library_instance_results[library].append(plausible_cases)

    library_instance_correct_results = {}
    for instance, correct_cases in instance_correct_results.items():
        library = instance.rsplit('_', 1)[0] if '_' in instance else instance
        if library not in library_instance_correct_results:
            library_instance_correct_results[library] = []
        library_instance_correct_results[library].append(correct_cases)
    
    plause_at_k_per_library = {}
    for library, plausible_cases_list in library_instance_results.items():
        lib_total = len(plausible_cases_list)
        lib_passed_any = sum(1 for s in plausible_cases_list if any(s))
        lib_passed_all = sum(1 for s in plausible_cases_list if all(s))
        plause_at_k_per_library[library] = {
            "pass_at_k_count": lib_passed_any,
            "pass_at_k_rate": lib_passed_any / lib_total if lib_total else 0.0,
            "pass_all_k_count": lib_passed_all,
            "pass_all_k_rate": lib_passed_all / lib_total if lib_total else 0.0,
            "total_instances": lib_total
        }

    correct_at_k_per_library = {}
    for library, correct_cases_list in library_instance_correct_results.items():
        lib_total = len(correct_cases_list)
        lib_correct_any = sum(1 for s in correct_cases_list if any(s))
        lib_correct_all = sum(1 for s in correct_cases_list if all(s))
        correct_at_k_per_library[library] = {
            "pass_at_k_count": lib_correct_any,
            "pass_at_k_rate": lib_correct_any / lib_total if lib_total else 0.0,
            "pass_all_k_count": lib_correct_all,
            "pass_all_k_rate": lib_correct_all / lib_total if lib_total else 0.0,
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

    costs = [r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in results]
    exec_times = [r.get('execution_time_seconds', 0.0) for r in results]
    steps = [r.get('total_steps', 0) for r in results]
    failed_instances = [f"run_{r['run']}/{r['instance']}" for r in results if not is_plausible_result(r)]
    plausible_instances = [f"run_{r['run']}/{r['instance']}" for r in results if is_plausible_result(r)]
    incorrect_instances = [f"run_{r['run']}/{r['instance']}" for r in results if not is_correct_result(r)]
    correct_instances = [f"run_{r['run']}/{r['instance']}" for r in results if is_correct_result(r)]
    
    # Per-run statistics
    per_run_stats = {}
    for run_num in sorted(run_numbers):
        run_results = [r for r in results if r.get('run') == run_num]
        if not run_results:
            continue
        
        run_costs = [r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in run_results]
        run_exec_times = [r.get('execution_time_seconds', 0.0) for r in run_results]
        run_steps = [r.get('total_steps', 0) for r in run_results]
        run_plausible = sum(1 for r in run_results if is_plausible_result(r))
        run_correct = sum(1 for r in run_results if is_correct_result(r))
        run_incorrect = len(run_results) - run_correct
        
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
            "plausible_rate": run_plausible / len(run_results) if run_results else 0.0,
            "correct_rate": run_correct / len(run_results) if run_results else 0.0,
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
                "avg_cost_per_correct": sum(r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in run_results if is_correct_result(r)) / run_correct if run_correct else 0.0,
                "avg_cost_per_incorrect": sum(r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in run_results if not is_correct_result(r)) / run_incorrect if run_incorrect else 0.0,
            },
            "execution_time_distribution": {
                "total_execution_time_seconds": sum(run_exec_times),
                "avg_execution_time_seconds": sum(run_exec_times) / len(run_results) if run_results else 0.0,
                "max_execution_time_seconds": max(run_exec_times) if run_exec_times else 0.0,
                "min_execution_time_seconds": min(run_exec_times) if run_exec_times else 0.0,
                "median_execution_time_seconds": median(run_exec_times),
                "avg_execution_time_per_correct": sum(r.get('execution_time_seconds', 0.0) for r in run_results if is_correct_result(r)) / run_correct if run_correct else 0.0,
                "avg_execution_time_per_incorrect": sum(r.get('execution_time_seconds', 0.0) for r in run_results if not is_correct_result(r)) / run_incorrect if run_incorrect else 0.0,
            },
            "steps_distribution": {
                "total_steps": sum(run_steps),
                "avg_steps": sum(run_steps) / len(run_results) if run_results else 0,
                "max_steps": max(run_steps) if run_steps else 0,
                "min_steps": min(run_steps) if run_steps else 0,
                "median_steps": median(run_steps),
                "avg_steps_per_correct": sum(r.get('total_steps', 0) for r in run_results if is_correct_result(r)) / run_correct if run_correct else 0,
                "avg_steps_per_incorrect": sum(r.get('total_steps', 0) for r in run_results if not is_correct_result(r)) / run_incorrect if run_incorrect else 0,
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
        "plausible_outcome_distribution": {
            "pass_at_k_rate": plause_at_k_rate,
            "pass_at_k_count": plause_at_k_count,
            "pass_all_k_rate": plause_all_k_rate,
            "pass_all_k_count": plause_all_k_count,
            "per_run": {run: {"plausible_rate": per_run_stats[run]["plausible_rate"]} for run in sorted(per_run_stats.keys())},
            "pass_at_k_per_library": plause_at_k_per_library,
            "failed_instances": failed_instances,
            "plausible_instances": plausible_instances,
        },
        "correct_outcome_distribution": {
            "pass_at_k_rate": correct_at_k_rate,
            "pass_at_k_count": correct_at_k_count,
            "pass_all_k_rate": correct_all_k_rate,
            "pass_all_k_count": correct_all_k_count,
            "per_run": {run: {"correct_rate": per_run_stats[run]["correct_rate"]} for run in sorted(per_run_stats.keys())},
            "pass_at_k_per_library": correct_at_k_per_library,
            "incorrect_instances": incorrect_instances,
            "correct_instances": correct_instances,
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
                "avg_cost_per_plausible": sum(r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in results if is_plausible_result(r)) / plausible if plausible else 0.0,
                "avg_cost_per_failure": sum(r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in results if not is_plausible_result(r)) / failed if failed else 0.0,
                "avg_cost_per_correct": sum(r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in results if is_correct_result(r)) / correct if correct else 0.0,
                "avg_cost_per_incorrect": sum(r.get('cost', 0.0) if r.get('cost', 0.0) else 0.0 for r in results if not is_correct_result(r)) / incorrect if incorrect else 0.0,
            },
            "per_run": {run: per_run_stats[run]["cost_distribution"] for run in sorted(per_run_stats.keys())}
        },
        "execution_time_distribution": {
            "overall": {
                "avg_execution_time_seconds": total_execution_time_seconds / len(results) if results else 0.0,
                "max_execution_time_seconds": max(exec_times) if exec_times else 0.0,
                "min_execution_time_seconds": min(exec_times) if exec_times else 0.0,
                "median_execution_time_seconds": median(exec_times),
                "avg_execution_time_per_plausible": sum(r.get('execution_time_seconds', 0.0) for r in results if is_plausible_result(r)) / plausible if plausible else 0.0,
                "avg_execution_time_per_failure": sum(r.get('execution_time_seconds', 0.0) for r in results if not is_plausible_result(r)) / failed if failed else 0.0,
                "avg_execution_time_per_correct": sum(r.get('execution_time_seconds', 0.0) for r in results if is_correct_result(r)) / correct if correct else 0.0,
                "avg_execution_time_per_incorrect": sum(r.get('execution_time_seconds', 0.0) for r in results if not is_correct_result(r)) / incorrect if incorrect else 0.0,
            },
            "per_run": {run: per_run_stats[run]["execution_time_distribution"] for run in sorted(per_run_stats.keys())}
        },
        "steps_distribution": {
            "overall": {
                "avg_steps": total_steps / len(results) if results else 0,
                "max_steps": max(steps) if steps else 0,
                "min_steps": min(steps) if steps else 0,
                "median_steps": median(steps),
                "avg_steps_per_plausible": sum(r.get('total_steps', 0) for r in results if is_plausible_result(r)) / plausible if plausible else 0,
                "avg_steps_per_failure": sum(r.get('total_steps', 0) for r in results if not is_plausible_result(r)) / failed if failed else 0,
                "avg_steps_per_correct": sum(r.get('total_steps', 0) for r in results if is_correct_result(r)) / correct if correct else 0,
                "avg_steps_per_incorrect": sum(r.get('total_steps', 0) for r in results if not is_correct_result(r)) / incorrect if incorrect else 0,
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
            "total_plausible": plausible,
            "total_correct": correct,
            # "total_plausible_pass_at_k": plause_at_k_count,
            # "total_plausible_pass_all_k": pass_all_k_count,
            # "total_failed": failed,
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

def main(base_dir: Path):
    if not base_dir.exists():
        print(f"Skipping {base_dir} (does not exist)")
        return

    # Collect results directly from the base path
    all_results = collect_results(base_dir)
    
    if not all_results:
        print(f"No results found in {base_dir}")
        return
    
    output_path = base_dir / "overall_summary.json"
    generate_overall_summary(all_results, output_path)

#-------------------------------------------------

def _exact_mcnemar_p_value(n10: int, n01: int) -> float:
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    lower_tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower_tail)

def _instance_pass_at_k_from_results(results: list[dict]) -> dict[str, int]:
    by_instance: dict[str, list[bool]] = {}
    for r in results:
        instance = r.get("instance", "")
        by_instance.setdefault(instance, []).append(bool(r.get("is_correct", False)))
    return {
        instance: int(sum(run_correctness) >= 1)
        for instance, run_correctness in by_instance.items()
    }

def compare_settings_pass_at_k(
    setting_to_results: dict[str, list[dict]],
    alpha: float = 0.05,
) -> dict[str, object]:
    comparisons: list[dict] = []

    for setting_a, setting_b in combinations(sorted(setting_to_results.keys()), 2):
        outcomes_a = _instance_pass_at_k_from_results(setting_to_results[setting_a])
        outcomes_b = _instance_pass_at_k_from_results(setting_to_results[setting_b])

        common_instances = sorted(set(outcomes_a.keys()) & set(outcomes_b.keys()))

        n11 = n10 = n01 = n00 = 0
        for instance in common_instances:
            a = outcomes_a[instance]
            b = outcomes_b[instance]
            if a == 1 and b == 1:
                n11 += 1
            elif a == 1 and b == 0:
                n10 += 1
            elif a == 0 and b == 1:
                n01 += 1
            else:
                n00 += 1

        n_instances = len(common_instances)
        p_value = _exact_mcnemar_p_value(n10, n01)
        delta = ((n10 - n01) / n_instances) if n_instances else 0.0

        rates_a = _instance_correct_rate_from_results(setting_to_results[setting_a])
        rates_b = _instance_correct_rate_from_results(setting_to_results[setting_b])

        x = [rates_a[i] for i in common_instances]
        y = [rates_b[i] for i in common_instances]
        wilcoxon_result = _wilcoxon_signed_rank_p_value_paired(x, y)

        comparisons.append(
            {
                "setting_a": setting_a,
                "setting_b": setting_b,
                "n_instances": n_instances,
                "contingency_table": {
                    "n11_both_pass": n11,
                    "n10_a_pass_b_fail": n10,
                    "n01_a_fail_b_pass": n01,
                    "n00_both_fail": n00,
                },
                "discordant_pairs": n10 + n01,
                "delta_pass_rate_a_minus_b": delta,
                "mcnemar_exact_p_value": p_value,
                "significant_at_0_05": p_value < alpha,
                "per_instance_rate_wilcoxon": {
                    "method": "Wilcoxon signed-rank (two-sided, normal approximation, tie-corrected)",
                    "p_value_two_sided": wilcoxon_result["p_value_two_sided"],
                    "significant_at_0_05": wilcoxon_result["p_value_two_sided"] < alpha,
                    "n_pairs": wilcoxon_result["n_pairs"],
                    "n_nonzero": wilcoxon_result["n_nonzero"],
                    "w_plus": wilcoxon_result["w_plus"],
                    "w_minus": wilcoxon_result["w_minus"],
                    "w_stat": wilcoxon_result["w_stat"],
                    "z_stat": wilcoxon_result["z_stat"],
                    "mean_rate_difference_a_minus_b": wilcoxon_result["mean_difference"],
                    "median_rate_difference_a_minus_b": wilcoxon_result["median_difference"],
                },
            }
        )

    return {
        "alpha": alpha,
        "method": "Exact McNemar (two-sided), paired by instance pass@k outcomes",
        "comparisons": comparisons,
    }

def collect_results_for_setting_model(
    results_root: Path,
    model: str,
    settings: list[str],
) -> dict[str, list[dict]]:
    setting_to_results: dict[str, list[dict]] = {}
    for setting in settings:
        run_dir = results_root / setting / model
        if run_dir.exists():
            setting_to_results[setting] = collect_results(run_dir)
    return setting_to_results

def run_passk_pairwise_significance(
    results_root: Path,
    model: str,
    settings: list[str],
    output_path: Path,
) -> None:
    setting_to_results = collect_results_for_setting_model(results_root, model, settings)
    report = compare_settings_pass_at_k(setting_to_results=setting_to_results, alpha=0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved pairwise pass@k significance: {output_path}")

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _average_ranks(values: list[float]) -> list[float]:
    # Average ranks for ties, ranks start at 1.
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    rank = 1
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        rank += (j - i)
        i = j
    return ranks


def _instance_correct_rate_from_results(results: list[dict]) -> dict[str, float]:
    by_instance: dict[str, list[int]] = {}
    for r in results:
        instance = r.get("instance", "")
        by_instance.setdefault(instance, []).append(int(bool(r.get("is_correct", False))))
    return {
        instance: (sum(vals) / len(vals) if vals else 0.0)
        for instance, vals in by_instance.items()
    }


def _wilcoxon_signed_rank_p_value_paired(
    x: list[float],
    y: list[float],
) -> dict[str, float | int]:
    # Wilcoxon signed-rank, two-sided, normal approximation with tie correction.
    diffs = [a - b for a, b in zip(x, y)]
    nonzero = [d for d in diffs if d != 0.0]
    n_pairs = len(diffs)
    n_nonzero = len(nonzero)

    if n_nonzero == 0:
        return {
            "n_pairs": n_pairs,
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "w_stat": 0.0,
            "z_stat": 0.0,
            "p_value_two_sided": 1.0,
            "mean_difference": 0.0,
            "median_difference": 0.0,
        }

    abs_vals = [round(abs(d), 12) for d in nonzero]
    ranks = _average_ranks(abs_vals)

    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w_stat = min(w_plus, w_minus)

    n = n_nonzero
    tie_counts: dict[float, int] = {}
    for v in abs_vals:
        tie_counts[v] = tie_counts.get(v, 0) + 1

    tie_term = sum(t * (t + 1) * (2 * t + 1) for t in tie_counts.values() if t > 1)
    var_w_plus = (n * (n + 1) * (2 * n + 1) - tie_term) / 24.0
    mean_w_plus = n * (n + 1) / 4.0

    if var_w_plus <= 0:
        z = 0.0
        p_value = 1.0
    else:
        # continuity correction
        z = (abs(w_plus - mean_w_plus) - 0.5) / sqrt(var_w_plus)
        p_value = max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z)))))

    sorted_d = sorted(nonzero)
    m = len(sorted_d)
    median_d = sorted_d[m // 2] if m % 2 == 1 else (sorted_d[m // 2 - 1] + sorted_d[m // 2]) / 2.0
    mean_d = sum(nonzero) / len(nonzero)

    return {
        "n_pairs": n_pairs,
        "n_nonzero": n_nonzero,
        "w_plus": float(w_plus),
        "w_minus": float(w_minus),
        "w_stat": float(w_stat),
        "z_stat": float(z),
        "p_value_two_sided": float(p_value),
        "mean_difference": float(mean_d),
        "median_difference": float(median_d),
    }