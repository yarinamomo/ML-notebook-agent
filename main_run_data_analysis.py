from pathlib import Path
from data_analysis import (
    edit_cell_analyze,
    find_early_submissions,
    # run_code_analyze,
    run_code_visualize,
    summarize_plots,
    summarize_statistics,
)


def main():
    target_modes = ["baseline"]  # , "agent", "without_run_code"
    target_models = ["glm-4.7-355b"]
    for target_mode in target_modes:
        for target_model in target_models:
            target_result_dir = Path(f"results/{target_mode}/{target_model}")

            summarize_statistics.main(target_result_dir)
            if "baseline" not in target_mode:
                summarize_plots.main(target_result_dir)
                find_early_submissions.main(target_result_dir, max_step=3)
            if "agent" in target_mode:
                # run_code_analyze.main(target_result_dir)
                run_code_visualize.main(target_result_dir)
            edit_cell_analyze.main(target_result_dir)

    # Create tool comparison chart per step across two settings: without "run_code" tool vs. default agent
    summarize_plots.create_comparison_chart(
        "results/without_run_code/glm-4.7-355b",
        "results/agent/glm-4.7-355b",
        'Tool Count Comparison - Without "run_code" Tool vs. Default Agent (glm-4.7-355b)',
        Path("results/data_analysis/tool_comparison_agents_without_run_code.png"),
        label1='Agent without "run_code" tool',
        label2="Agent default",
        mode="diff",
    )
    summarize_plots.create_comparison_chart(
        "results/without_run_code/glm-4.7-355b",
        "results/agent/glm-4.7-355b",
        'Tool Count Comparison - Without "run_code" Tool vs. Default Agent (glm-4.7-355b)',
        Path("results/data_analysis/tool_comparison_agents_without_run_code.png"),
        label1='Agent without "run_code" tool',
        label2="Agent default",
        mode="full",
    )
    summarize_plots.compare_performance_across_settings(
        settings=["baseline", "without_run_code", "agent"]
    )


if __name__ == "__main__":
    main()
