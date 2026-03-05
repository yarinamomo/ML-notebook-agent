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
    target_modes = ["baseline"] # , "agent", "without_run_code"
    target_models = ["glm-4.7-355b"]
    for target_mode in target_modes:
        for target_model in target_models:
            target_result_dir = Path(f"results/{target_mode}/{target_model}")

            summarize_statistics.main(target_result_dir)
            if target_mode != "baseline":
                summarize_plots.main(target_result_dir)
                find_early_submissions.main(target_result_dir, max_step=3)
            if target_mode == "agent":
                # run_code_analyze.main(target_result_dir)
                run_code_visualize.main(target_result_dir)
            edit_cell_analyze.main(target_result_dir)

if __name__ == "__main__":
    main()