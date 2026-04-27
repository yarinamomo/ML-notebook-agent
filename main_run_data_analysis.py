from pathlib import Path
from data_analysis import (
    edit_cell_analyze,
    find_early_submissions,
    # run_code_analyze,
    run_code_visualize,
    summarize_plots,
    summarize_statistics,
    compare_fixed_notebooks,
)

def main():
    results_dir = Path("results")
    target_modes = ["baseline", "baseline_without_all_outputs", "baseline_without_cell_outputs", "agent", "agent_without_run_code_and_cell_outputs"]
    target_models = ["glm-4.7-355b"]
    for target_mode in target_modes:
        for target_model in target_models:
            target_result_dir = results_dir / target_mode / target_model

            summarize_statistics.main(target_result_dir)
            if "baseline" not in target_mode:
                summarize_plots.main(target_result_dir)
                find_early_submissions.main(target_result_dir, max_step=2)
            if target_mode == "agent":
                # run_code_analyze.main(target_result_dir)
                run_code_visualize.main(target_result_dir)
            edit_cell_analyze.main(target_result_dir)

    compare_fixed_notebooks.main()

    # Create tool comparison chart per step across two settings: without "run_code" tool vs. default agent
    summarize_plots.create_comparison_chart(
        results_dir / "agent_without_run_code_and_cell_outputs" / "glm-4.7-355b",
        results_dir / "agent" / "glm-4.7-355b",
        "Tool Count Comparison - Without \"run_code\" Tool vs. Default Agent (glm-4.7-355b)",
        results_dir / "data_analysis" / "tool_comparison_agents_without_run_code.png",
        label1="Agent without \"run_code\" tool",
        label2="Agent default",
        mode="diff"
    )
    summarize_plots.create_comparison_chart(
        results_dir / "agent_without_run_code_and_cell_outputs" / "glm-4.7-355b",
        results_dir / "agent" / "glm-4.7-355b",
        "Tool Count Comparison - Without \"run_code\" Tool vs. Default Agent (glm-4.7-355b)",
        results_dir / "data_analysis" / "tool_comparison_agents_without_run_code.png",
        label1="Agent without \"run_code\" tool",
        label2="Agent default",
        mode="full"
    )

    summarize_plots.compare_performance_across_settings(
        results_dir = results_dir,
        settings=target_modes
    )

    for target_model in target_models:
        summarize_statistics.run_passk_pairwise_significance(
            results_root=results_dir,
            model=target_model,
            settings=["agent", "agent_without_run_code_and_cell_outputs", "baseline"],
            k=1,
            output_path=results_dir / Path(f"data_analysis/passk_pairwise_significance_{target_model}.json"),
        )

if __name__ == "__main__":
    main()