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
    target_model = "gpt-5.3-codex" #"glm-4.7-355b"
    for target_mode in target_modes:
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

    # # Create tool comparison chart per step across two settings: without "run_code" tool vs. default agent
    # summarize_plots.create_comparison_chart(
    #     results_dir / "agent_without_run_code_and_cell_outputs" / target_model,
    #     results_dir / "agent" / target_model,
    #     "Tool Count Comparison - Without \"run_code\" Tool vs. Default Agent (" + target_model + ")",
    #     results_dir / "data_analysis" / f"tool_comparison_agents_without_run_code_{target_model}.png",
    #     label1="Agent without \"run_code\" tool",
    #     label2="Agent default",
    #     mode="diff"
    # )
    # summarize_plots.create_comparison_chart(
    #     results_dir / "agent_without_run_code_and_cell_outputs" / target_model,
    #     results_dir / "agent" / target_model,
    #     "Tool Count Comparison - Without \"run_code\" Tool vs. Default Agent (" + target_model + ")",
    #     results_dir / "data_analysis" / f"tool_comparison_agents_without_run_code_{target_model}.png",
    #     label1="Agent without \"run_code\" tool",
    #     label2="Agent default",
    #     mode="full"
    # )

    # # Create aggregated category proportion comparison between settings
    # summarize_plots.create_comparison_chart_agg(
    #     results_dir / "agent_without_run_code_and_cell_outputs" / target_model,
    #     results_dir / "agent" / target_model,
    #     "Tool Category Proportions (" + target_model + ")",
    #     results_dir / "data_analysis" / f"tool_category_proportions_comparison_{target_model}.png",
    #     label1="Agent-lite",
    #     label2="Agent-full",
    #     error_bars="ci",
    #     save_pdf=True
    # )

    target_models = ["gpt-5.3-codex", "glm-4.7-355b"]
    for target_model in target_models:
        summarize_plots.create_comparison_chart_agg_by_outcome(
            results_dir / "agent_without_run_code_and_cell_outputs" / target_model,
            results_dir / "agent" / target_model,
            "Tool Category Proportions by Outcome (" + target_model + ")",
            results_dir / "data_analysis" / f"tool_category_proportions_by_outcome_comparison_{target_model}.png",
            label1="Agent-lite",
            label2="Agent-full",
            error_bars="ci",
            save_pdf=True
        )

    target_models = ["gpt-5.3-codex", "glm-4.7-355b"]
    for target_model in target_models:
        summarize_plots.create_run_code_usage_plot(
            results_dir / "agent" / target_model,
            "run_code Usage Summary (" + target_model + ")",
            results_dir / "data_analysis" / f"run_code_usage_summary_{target_model}.png",
            label="Agent-full",
            save_pdf=False
        )
        summarize_plots.create_run_code_usage_plot_by_outcome(
            results_dir / "agent" / target_model,
            "run_code Usage Summary by Outcome (" + target_model + ")",
            results_dir / "data_analysis" / f"run_code_usage_summary_by_outcome_{target_model}.png",
            label="Agent-full",
            save_pdf=False
        )
        summarize_plots.print_run_code_usage_table(
            results_dir / "agent" / target_model
        )
    
    summarize_plots.compare_pass_at_k_across_bug_types(
        bug_category="library",
        bug_categories=['tensorflow', 'torch', 'sklearn', 'numpy', 'pandas', 'visual', 'minor'],
        settings = ['baseline_without_cell_outputs', 'agent_without_run_code_and_cell_outputs', 'agent'],
        category_grouping={
            'visual': ['seaborn', 'matplotlib'], 
            'minor': ['statsmodels', 'torchvision', 'lightgbm']},
        save_pdf=True)
    # summarize_plots.compare_pass_at_k_across_bug_types(
    #     bug_category="root_cause",
    #     bug_categories=['API misuse', 'data confusion', 'implementation error'],
    #     settings = ['baseline_without_cell_outputs', 'agent_without_run_code_and_cell_outputs', 'agent'],
    #     category_grouping={'other': ['ML model confusion', 'library cause']}
    # )
    summarize_plots.compare_pass_at_k_across_bug_types(
        bug_category="pipeline",
        bug_categories=['data process', 'data visual', 'model construct', 'model train', 'model infer'],
        settings = ['baseline_without_cell_outputs', 'agent_without_run_code_and_cell_outputs', 'agent'],
        category_grouping={
            'model train': ['training'], 'model infer': ['evaluation/prediction'], 
            'model construct': ['model construction'], 'data visual': ['data visualization'], 
            'data process': ['data preparation']},
        save_pdf=True
    )
    summarize_plots.compare_pass_at_k_across_bug_types(
        bug_category="crash_type",
        settings=['baseline_without_cell_outputs', 'agent_without_run_code_and_cell_outputs', 'agent'],
        bug_categories=['data mismatch', 'invalid arg', 'value error', 'attribute error', 'key error', 'index error', 'other'],
        category_grouping={
            'invalid arg': ['invalid argument'],
            'data mismatch': ['tensor shape mismatch', 'data value violation', 'unsupported broadcast', 'feature name mismatch'],
            'other': ['runtime error', 'io error', 'model initialization error', 'type error']},
        save_pdf=True)

    summarize_plots.compare_performance_across_settings(
        results_dir = results_dir,
        settings=target_modes
    )

    summarize_statistics.run_passk_pairwise_significance(
        results_root=results_dir,
        model=target_model,
        settings=target_modes, #["agent", "agent_without_run_code_and_cell_outputs", "baseline"],
        output_path=results_dir / Path(f"data_analysis/passk_pairwise_significance_{target_model}.json"),
    )

if __name__ == "__main__":
    main()