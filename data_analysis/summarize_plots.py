"""
Summarize and visualize operations taken by LLM runs across different test cases.

This script generates stacked bar charts showing the count of different operations/actions
taken by the LLM agent at each step of execution.
"""

import json
import textwrap
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from data_analysis.analyze_patched_executed_failures import evaluate_notebook_correctness

# Define the valid tool names from notebook_tools._TOOL_PARAM_ORDER
# Ordered for stacked bar chart display
VALID_TOOLS = [
    "run_all",
    "run_cell",
    "get_cells",
    "get_cell",
    "get_cell_count",
    "edit_cell",
    "run_code",
    "Submitted",
    "submit",
]

# Fixed colors per tool for consistent appearance across settings
_tab10 = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
_tab20 = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
TOOL_COLORS = {
    "run_all":       _tab10[0],
    "run_cell":      _tab10[1],
    "get_cells":     _tab10[2],
    "get_cell":      _tab10[3],
    "get_cell_count": _tab10[4],
    "edit_cell":     _tab10[5],
    "run_code":      _tab10[6],
    "Submitted":     _tab10[7],
    "submit":        _tab10[7],
}

SETTING_COLORS = {
    "baseline_without_cell_outputs":       _tab20[5],
    "agent_without_run_code_and_cell_outputs": _tab20[1],
    "agent":           _tab20[3],
}

CATEGORY_ORDER = ["Inspect", "Execute", "Edit", "Submit"]
CATEGORY_TOOL_MAP = {
    "Inspect": {"get_cell", "get_cells", "get_cell_count"},
    "Execute": {"run_cell", "run_all"},
    "Edit": {"edit_cell"},
    "Submit": {"submit", "Submitted"},
}


LLM_ABBREVIATIONS = {
    'glm-4.7-355b': 'GLM'
}

SETTING_LABEL_MAP = {
    'baseline': 'SS+CO',
    'baseline_without_cell_outputs': 'SS',
    'baseline_without_all_outputs': 'SS-ERR',
    'agent_without_run_code_and_cell_outputs': 'Agent-lite',
    'agent': 'Agent-full',
}

BUG_CATEGORY_COLUMN_MAP = {
    'library': ('nb_name', 'Library'),
    'root_cause': ('label_root_cause', 'Root Cause'),
    'crash_type': ('label_refined_exp_type', 'Crash Type'),
    'pipeline': ('label_ML_pipeline', 'Pipeline Stage'),
}

def normalize_tool_name(tool_name):
    """Normalize synonymous tool names to a shared canonical form."""
    if tool_name == "Submitted":
        return "submit"
    return tool_name


def get_tool_category(tool_name):
    """Map a tool name to one of the high-level behavior categories."""
    normalized_tool_name = normalize_tool_name(tool_name)

    for category, tool_names in CATEGORY_TOOL_MAP.items():
        normalized_tool_names = {normalize_tool_name(name) for name in tool_names}
        if normalized_tool_name in normalized_tool_names:
            return category

    return None


def save_figure(fig, output_path: Path, save_pdf: bool = False):
    """Save a matplotlib figure to the requested image path and optionally to PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')

    print(f"Saved figure to {output_path}")
    if save_pdf:
        print(f"Saved figure to {output_path.with_suffix('.pdf')}")


def extract_tool_name(action_string):
    """
    Extract the tool name from an action string.
    
    Examples:
        "run_all()" -> "run_all"
        "edit_cell(cell_index=3, code=<435 chars>)" -> "edit_cell"
        "get_cells()" -> "get_cells"
        "LimitsExceeded" -> None (not a valid tool)
    
    Args:
        action_string: The action string from operations
    
    Returns:
        Tool name if valid, None otherwise
    """
    if not action_string or not isinstance(action_string, str):
        return None
    
    # Extract the function name before the first parenthesis
    if '(' in action_string:
        tool_name = action_string.split('(')[0]
    else:
        tool_name = action_string
    
    # Only return if it's a valid tool
    if tool_name in VALID_TOOLS:
        return tool_name
    
    return None


def load_summary_data(run_dir):
    """
    Load all summary JSON files from a run directory.
    
    Args:
        run_dir: Path to the run directory
    
    Returns:
        Dictionary mapping instance_id to parsed data
    """
    run_path = Path(run_dir)
    data = {}
    
    # Find all *_summary.json files
    for summary_file in run_path.glob('*_summary.json'):
        instance_id = summary_file.stem.replace('_summary', '')
        
        try:
            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                summary = json.load(f)
                
            # Extract operations with valid tool names
            operations_by_step = defaultdict(lambda: defaultdict(int))
            
            if 'operations' in summary:
                for op in summary['operations']:
                    step = op.get('step')
                    action = op.get('action')
                    
                    if step is not None and action:
                        tool_name = extract_tool_name(action)
                        if tool_name:
                            operations_by_step[step][tool_name] += 1
            
            data[instance_id] = operations_by_step
            
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading {summary_file}: {e}")
    
    return data


def aggregate_operations(data_dict):
    """
    Aggregate operations across all test cases in a run.
    
    Args:
        data_dict: Dictionary of instance_id -> operations_by_step
    
    Returns:
        Dictionary mapping step -> tool_name -> count
    """
    aggregated = defaultdict(lambda: defaultdict(int))
    
    for _instance_id, operations_by_step in data_dict.items():
        for step, tool_counts in operations_by_step.items():
            for tool_name, count in tool_counts.items():
                aggregated[step][tool_name] += count
    
    return aggregated


def create_stacked_bar_chart(aggregated_data, title, output_path, save_pdf: bool = False):
    """
    Create a stacked bar chart for operations by step.

    Args:
        aggregated_data: Dictionary mapping step -> tool_name -> count
        title: Title for the chart
        output_path: Path to save the chart
    """
    if not aggregated_data:
        print(f"No data to plot for {title}")
        return

    steps = sorted(aggregated_data.keys())

    all_tools = set()
    for tool_counts in aggregated_data.values():
        all_tools.update(tool_counts.keys())

    tools = [tool for tool in VALID_TOOLS if tool in all_tools]
    tools.extend(sorted([tool for tool in all_tools if tool not in VALID_TOOLS]))

    data_matrix = np.zeros((len(tools), len(steps)))
    for i, step in enumerate(steps):
        for j, tool in enumerate(tools):
            data_matrix[j, i] = aggregated_data[step].get(tool, 0)

    _fig, ax = plt.subplots(figsize=(max(12, len(steps) * 0.5), 8))

    bottom = np.zeros(len(steps))
    for j, tool in enumerate(tools):
        color = TOOL_COLORS.get(tool, plt.get_cmap('tab10')((len(TOOL_COLORS) + list(tools).index(tool)) / 10))
        ax.bar(range(len(steps)), data_matrix[j], bottom=bottom, label=tool, color=color, edgecolor='white', linewidth=0.5)
        bottom += data_matrix[j]

    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operation Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45 if len(steps) > 20 else 0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_figure(_fig, output_path, save_pdf=save_pdf)
    plt.close()

def _aggregate_notebook_category_proportions_by_outcome(path):
    """Aggregate run-level category proportions grouped by outcome label.

    Returns a dict mapping normalized outcome keys ('correct', 'failed') to a
    dict of instance_id -> mean_vector (averaged across runs with that outcome).
    """
    path_obj = Path(path)
    # For each instance, collect a list of (outcome, proportion_vector)
    notebook_runs = defaultdict(list)

    for run_dir in _iter_run_directories(path_obj):
        run_path = Path(run_dir)

        for summary_file in run_path.glob('*_summary.json'):
            instance_id = summary_file.stem.replace('_summary', '')
            try:
                with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                    summary = json.load(f)

                # Build category proportions for this single run/instance
                category_counts = defaultdict(int)
                total_tool_calls = 0

                for op in summary.get('operations', []):
                    action = op.get('action')
                    tool_name = extract_tool_name(action)
                    if not tool_name:
                        continue
                    total_tool_calls += 1
                    normalized_tool_name = normalize_tool_name(tool_name)
                    category = get_tool_category(normalized_tool_name)
                    if category:
                        category_counts[category] += 1

                if total_tool_calls == 0:
                    continue

                proportions = {
                    category: category_counts.get(category, 0) / total_tool_calls
                    for category in CATEGORY_ORDER
                }

                # Require the patched-executed notebook to determine correctness.
                notebook_ipynb = run_path / f"{instance_id}_patched_executed.ipynb"
                is_correct, _, _, _ = evaluate_notebook_correctness(notebook_ipynb)
                outcome = 'correct' if is_correct else 'failed'
                notebook_runs[instance_id].append((outcome, proportions))

            except (OSError, json.JSONDecodeError) as e:
                print(f"Error loading {summary_file}: {e}")

    # Now compute per-instance mean vectors separately for 'correct' and 'failed'
    outcome_groups = {'correct': {}, 'failed': {}}

    for instance_id, runs in notebook_runs.items():
        # group by outcome
        by_outcome = defaultdict(list)
        for outcome, prop in runs:
            vec = [prop.get(category, 0.0) for category in CATEGORY_ORDER]
            by_outcome[outcome].append(vec)

        for olabel, vecs in by_outcome.items():
            if not vecs:
                continue
            matrix = np.array(vecs, dtype=float)
            outcome_groups[olabel][instance_id] = matrix.mean(axis=0)

    return outcome_groups


def create_comparison_chart_agg_by_outcome(path1: str, path2: str, title: str, output_path: Path,
                                          label1: str='Setting 1', label2: str='Setting 2',
                                          error_bars: str='ci', save_pdf: bool = False):
    """Create grouped bar chart comparing category proportions split by outcome.

    For each category on the x-axis, four bars are shown (left to right):
        - correct, setting1
        - correct, setting2
        - failed,  setting1
        - failed,  setting2

    The function attempts to extract per-run outcome labels from each
    run's *_summary.json using common keys; runs without an explicit
    correct/failed label are ignored.
    """
    font_scale = 1.5

    print(f"Loading data from {path1}...")
    groups1 = _aggregate_notebook_category_proportions_by_outcome(path1)

    print(f"Loading data from {path2}...")
    groups2 = _aggregate_notebook_category_proportions_by_outcome(path2)

    stats = {}
    for label, groups in (('s1', groups1), ('s2', groups2)):
        stats[label] = {}
        for outcome in ('correct', 'failed'):
            stats[label][outcome] = _summarize_category_statistics(groups.get(outcome, {}))

    # If both settings have no data for both outcomes, nothing to plot
    if all(stats[l][o] is None for l in stats for o in ('correct', 'failed')):
        print(f"Insufficient labeled data to plot {title}")
        return

    x = np.arange(len(CATEGORY_ORDER))
    bar_width = 0.18
    fig, ax = plt.subplots(figsize=(8, 6))

    # Helper to get means and yerr (in percent) for a stats object
    def _means_yerr_pct(s):
        if not s:
            return np.zeros(len(CATEGORY_ORDER)), np.zeros(len(CATEGORY_ORDER))
        means = s['means'] * 100
        yerr = (s['standard_error'] if error_bars == 'stderr' else s['confidence_interval']) * 100
        return means, yerr

    m1c, e1c = _means_yerr_pct(stats['s1']['correct'])
    m2c, e2c = _means_yerr_pct(stats['s2']['correct'])
    m1f, e1f = _means_yerr_pct(stats['s1']['failed'])
    m2f, e2f = _means_yerr_pct(stats['s2']['failed'])

    # Colors for the four bar groups
    c1 = _tab20[0]
    c2 = _tab20[1]
    c3 = _tab20[2]
    c4 = _tab20[3]

    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    bars = []
    bars.append(ax.bar(x + offsets[0], m1c, bar_width, yerr=e1c, capsize=4, label=f"{label1} success", color=c1, edgecolor='white'))
    bars.append(ax.bar(x + offsets[1], m1f, bar_width, yerr=e1f, capsize=4, label=f"{label1} failed", color=c2, edgecolor='white'))
    bars.append(ax.bar(x + offsets[2], m2c, bar_width, yerr=e2c, capsize=4, label=f"{label2} success", color=c3, edgecolor='white'))
    bars.append(ax.bar(x + offsets[3], m2f, bar_width, yerr=e2f, capsize=4, label=f"{label2} failed", color=c4, edgecolor='white'))

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER, fontsize=11 * font_scale)
    y_max = max(
        float(np.max(m1c + e1c)) if m1c.size else 0,
        float(np.max(m2c + e2c)) if m2c.size else 0,
        float(np.max(m1f + e1f)) if m1f.size else 0,
        float(np.max(m2f + e2f)) if m2f.size else 0,
        50.0,
    )
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_ylabel('Mean percentage of actions', fontsize=12 * font_scale, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Reduce the colored handle width in the legend: lower `handlelength` to make the
    # color swatch narrower, and reduce `handletextpad` to tighten text spacing.
    ax.legend(fontsize=9 * font_scale, handlelength=0.8, handletextpad=0.6, handleheight=0.8)

    # plt.title(title, fontsize=14 * font_scale, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, output_path, save_pdf=save_pdf)
    plt.close()


def aggregate_directory(path):
    """
    Load and aggregate summary data from a directory that may contain multiple runs.
    
    Args:
        path: Path to directory with runs (run_1, run_2, run_3) or single run directory
    
    Returns:
        Dictionary mapping step -> tool_name -> count aggregated across all runs
    """
    path_obj = Path(path)
    combined_data = defaultdict(lambda: defaultdict(int))
    
    # Check if this path has run_1, run_2, run_3 subdirectories
    runs_found = False
    for run_num in [1, 2, 3]:
        run_dir = path_obj / f'run_{run_num}'
        if run_dir.exists():
            runs_found = True
            print(f"  Loading from {run_dir}...")
            data = load_summary_data(run_dir)
            aggregated = aggregate_operations(data)
            
            # Combine into combined_data
            for step, tool_counts in aggregated.items():
                for tool_name, count in tool_counts.items():
                    combined_data[step][tool_name] += count
    
    # If no runs found, try loading directly from this path
    if not runs_found:
        print(f"  Loading from {path_obj}...")
        data = load_summary_data(path_obj)
        combined_data = aggregate_operations(data)
    
    return combined_data


def create_comparison_chart(path1: str, path2: str, title: str, output_path: Path, label1: str='Setting 1', label2: str='Setting 2', mode: str='diff', save_pdf: bool = False):
    """
    Create a comparison chart between two settings.
    
    Args:
        path1: Path to the first dataset (directory with runs or single run directory)
        path2: Path to the second dataset (directory with runs or single run directory)
        title: Title for the chart
        output_path: Path to save the chart
        label1: Label for the first setting
        label2: Label for the second setting
        mode: 'diff' for difference line chart (default), 'full' for grouped bar charts with subplots
    """
    print(f"Loading data from {path1}...")
    aggregated_data1 = aggregate_directory(path1)
    
    print(f"Loading data from {path2}...")
    aggregated_data2 = aggregate_directory(path2)

    if not aggregated_data1 or not aggregated_data2:
        print(f"Insufficient data to plot {title}")
        return

    # Get all steps that appear in either dataset
    all_steps = sorted(set(aggregated_data1.keys()) | set(aggregated_data2.keys()))
    
    # Get all tool names that appear in either dataset
    all_tools = set()
    for tool_counts in aggregated_data1.values():
        all_tools.update(tool_counts.keys())
    for tool_counts in aggregated_data2.values():
        all_tools.update(tool_counts.keys())
    
    # Sort tools using VALID_TOOLS order
    tools = [tool for tool in VALID_TOOLS if tool in all_tools]
    tools.extend(sorted([tool for tool in all_tools if tool not in VALID_TOOLS]))
    
    if mode == 'diff':
        # Plot differences on a single line chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot difference lines for each tool
        for j, tool in enumerate(tools):
            color = TOOL_COLORS.get(tool, plt.get_cmap('tab10')((len(TOOL_COLORS) + j) / 10))
            steps_list = list(all_steps)
            data1 = np.array([aggregated_data1.get(step, {}).get(tool, 0) for step in steps_list])
            data2 = np.array([aggregated_data2.get(step, {}).get(tool, 0) for step in steps_list])
            
            # Calculate difference (positive = more in setting 1, negative = more in setting 2)
            differences = data1 - data2
            
            # Plot line for this tool's difference
            ax.plot(range(len(steps_list)), differences, marker='o', linestyle='-', linewidth=2.5,
                   color=color, label=tool, alpha=0.8, markersize=6)
        
        # Add a horizontal line at y=0 to show the neutral point
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Customize the plot
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Difference', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(all_steps)))
        ax.set_xticklabels(all_steps, rotation=45 if len(all_steps) > 20 else 0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=1)
        ax.grid(True, alpha=0.3, linestyle='--')
        
    else:  # mode == 'full'
        # Plot grouped bars for each tool in subplots
        data_matrix1 = np.zeros((len(tools), len(all_steps)))
        data_matrix2 = np.zeros((len(tools), len(all_steps)))
        
        for i, step in enumerate(all_steps):
            for j, tool in enumerate(tools):
                data_matrix1[j, i] = aggregated_data1.get(step, {}).get(tool, 0)
                data_matrix2[j, i] = aggregated_data2.get(step, {}).get(tool, 0)
        
        # Create the plot with subplots for each tool
        num_tools = len(tools)
        num_cols = min(3, num_tools)
        num_rows = (num_tools + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
        if num_tools == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Calculate bar positions
        x = np.arange(len(all_steps))
        bar_width = 0.35
        
        # Plot each tool in its own subplot
        for j, tool in enumerate(tools):
            ax = axes[j]
            
            # Create grouped bars for this tool
            ax.bar(x - bar_width/2, data_matrix1[j], bar_width, label=label1, alpha=0.8)
            ax.bar(x + bar_width/2, data_matrix2[j], bar_width, label=label2, alpha=0.8)
            
            # Customize the subplot
            ax.set_xlabel('Step', fontsize=10, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax.set_title(tool, fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(all_steps, rotation=45 if len(all_steps) > 20 else 0, fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Hide unused subplots
        for idx in range(num_tools, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    # Append mode to filename before extension (e.g., comparison_diff.png)
    final_path = output_path.parent / (output_path.stem + f'_{mode}' + output_path.suffix)
    save_figure(fig, final_path, save_pdf=save_pdf)
    plt.close()


def _iter_run_directories(path_obj: Path):
    """Return run directories if present, otherwise the path itself as a single run."""
    run_directories = [path_obj / f'run_{run_num}' for run_num in [1, 2, 3] if (path_obj / f'run_{run_num}').exists()]
    return run_directories if run_directories else [path_obj]


def _load_notebook_run_metrics(run_dir: Path):
    """Load per-notebook category proportions and run_code usage for one run directory."""
    run_path = Path(run_dir)
    data = {}

    for summary_file in run_path.glob('*_summary.json'):
        instance_id = summary_file.stem.replace('_summary', '')

        try:
            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                summary = json.load(f)

            category_counts = defaultdict(int)
            total_tool_calls = 0
            run_code_count = 0

            for op in summary.get('operations', []):
                action = op.get('action')
                tool_name = extract_tool_name(action)

                if not tool_name:
                    continue

                total_tool_calls += 1

                normalized_tool_name = normalize_tool_name(tool_name)
                if normalized_tool_name == 'run_code':
                    run_code_count += 1

                category = get_tool_category(normalized_tool_name)
                if category:
                    category_counts[category] += 1

            if total_tool_calls > 0:
                data[instance_id] = {
                    'category_proportions': {
                        category: category_counts.get(category, 0) / total_tool_calls
                        for category in CATEGORY_ORDER
                    },
                    'run_code_count': run_code_count,
                    'total_tool_calls': total_tool_calls,
                }

        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading {summary_file}: {e}")

    return data


def _aggregate_notebook_category_proportions(path):
    """Average run-level category proportions within each notebook across its runs."""
    path_obj = Path(path)
    notebook_runs = defaultdict(list)

    for run_dir in _iter_run_directories(path_obj):
        run_metrics = _load_notebook_run_metrics(run_dir)
        for instance_id, metrics in run_metrics.items():
            notebook_runs[instance_id].append(metrics['category_proportions'])

    notebook_means = {}
    for instance_id, run_proportions in notebook_runs.items():
        if not run_proportions:
            continue

        matrix = np.array([
            [run_proportions_per_category.get(category, 0.0) for category in CATEGORY_ORDER]
            for run_proportions_per_category in run_proportions
        ], dtype=float)
        notebook_means[instance_id] = matrix.mean(axis=0)

    return notebook_means


def _summarize_category_statistics(notebook_means):
    """Compute mean and uncertainty across notebooks for each category."""
    if not notebook_means:
        return None

    matrix = np.array(list(notebook_means.values()), dtype=float)
    num_notebooks = matrix.shape[0]
    means = matrix.mean(axis=0)

    if num_notebooks > 1:
        standard_error = matrix.std(axis=0, ddof=1) / np.sqrt(num_notebooks)
    else:
        standard_error = np.zeros(matrix.shape[1], dtype=float)

    confidence_interval = 1.96 * standard_error

    return {
        'means': means,
        'standard_error': standard_error,
        'confidence_interval': confidence_interval,
        'num_notebooks': num_notebooks,
    }


def create_comparison_chart_agg(path1: str, path2: str, title: str, output_path: Path, label1: str='Setting 1', label2: str='Setting 2', error_bars: str='ci', save_pdf: bool = False):
    """Create a grouped bar chart comparing category proportions between two settings."""
    font_scale = 1.5

    print(f"Loading data from {path1}...")
    notebook_means1 = _aggregate_notebook_category_proportions(path1)

    print(f"Loading data from {path2}...")
    notebook_means2 = _aggregate_notebook_category_proportions(path2)

    stats1 = _summarize_category_statistics(notebook_means1)
    stats2 = _summarize_category_statistics(notebook_means2)

    if not stats1 or not stats2:
        print(f"Insufficient data to plot {title}")
        return

    x = np.arange(len(CATEGORY_ORDER))
    bar_width = 0.35
    _fig, ax = plt.subplots(figsize=(8, 6))

    if error_bars == 'stderr':
        yerr1 = stats1['standard_error']
        yerr2 = stats2['standard_error']
    else:
        yerr1 = stats1['confidence_interval']
        yerr2 = stats2['confidence_interval']

    means1_pct = stats1['means'] * 100
    means2_pct = stats2['means'] * 100
    yerr1_pct = yerr1 * 100
    yerr2_pct = yerr2 * 100

    color1 = _tab20[1]
    color2 = _tab20[3]

    ax.bar(x - bar_width / 2, means1_pct, bar_width, yerr=yerr1_pct, capsize=5,
           label=label1, color=color1, edgecolor='white', linewidth=0.8, alpha=0.95)
    ax.bar(x + bar_width / 2, means2_pct, bar_width, yerr=yerr2_pct, capsize=5,
           label=label2, color=color2, edgecolor='white', linewidth=0.8, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER, fontsize=11 * font_scale)
    y_max = max(float(np.max(means1_pct + yerr1_pct)), float(np.max(means2_pct + yerr2_pct)), 50.0)
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.set_ylabel('Mean percentage of actions', fontsize=12 * font_scale, fontweight='bold')
    # ax.set_title(title, fontsize=14 * font_scale, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10 * font_scale)

    plt.tight_layout()
    save_figure(_fig, output_path, save_pdf=save_pdf)
    plt.close()


def summarize_run_code_usage(path: str):
    """Summarize how often run_code appears across all runs in a path."""
    path_obj = Path(path)
    total_runs = 0
    runs_with_run_code = 0
    total_run_code_calls = 0

    for run_dir in _iter_run_directories(path_obj):
        run_metrics = _load_notebook_run_metrics(run_dir)
        for metrics in run_metrics.values():
            total_runs += 1
            total_run_code_calls += metrics['run_code_count']
            if metrics['run_code_count'] > 0:
                runs_with_run_code += 1

    if total_runs == 0:
        return None

    return {
        'total_runs': total_runs,
        'runs_with_run_code': runs_with_run_code,
        'run_code_usage_rate': runs_with_run_code / total_runs,
        'average_run_code_count_per_run': total_run_code_calls / total_runs,
    }


def _summarize_run_code_usage_by_outcome(path: str):
    """Summarize run_code usage separately for correct and failed runs.

    Correctness is determined only from evaluate_notebook_correctness on the
    corresponding *_patched_executed.ipynb file.
    """
    path_obj = Path(path)
    outcome_stats = {
        'correct': {'total_runs': 0, 'runs_with_run_code': 0, 'total_run_code_calls': 0},
        'failed': {'total_runs': 0, 'runs_with_run_code': 0, 'total_run_code_calls': 0},
    }

    for run_dir in _iter_run_directories(path_obj):
        run_path = Path(run_dir)

        for summary_file in run_path.glob('*_summary.json'):
            instance_id = summary_file.stem.replace('_summary', '')
            notebook_ipynb = run_path / f"{instance_id}_patched_executed.ipynb"
            is_correct, _, _, _ = evaluate_notebook_correctness(notebook_ipynb)

            outcome = 'correct' if is_correct else 'failed'

            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                summary = json.load(f)

            run_code_count = 0
            for op in summary.get('operations', []):
                action = op.get('action')
                tool_name = extract_tool_name(action)
                if not tool_name:
                    continue

                normalized_tool_name = normalize_tool_name(tool_name)
                if normalized_tool_name == 'run_code':
                    run_code_count += 1

            bucket = outcome_stats[outcome]
            bucket['total_runs'] += 1
            bucket['total_run_code_calls'] += run_code_count
            if run_code_count > 0:
                bucket['runs_with_run_code'] += 1

    for outcome, bucket in outcome_stats.items():
        total_runs = bucket['total_runs']
        if total_runs == 0:
            bucket['run_code_usage_rate'] = 0.0
            bucket['average_run_code_count_per_run'] = 0.0
        else:
            bucket['run_code_usage_rate'] = bucket['runs_with_run_code'] / total_runs
            bucket['average_run_code_count_per_run'] = bucket['total_run_code_calls'] / total_runs

    if outcome_stats['correct']['total_runs'] == 0 and outcome_stats['failed']['total_runs'] == 0:
        return None

    return outcome_stats


def create_run_code_usage_plot(path: str, title: str, output_path: Path, label: str='Setting', save_pdf: bool = False):
    """Create a compact plot summarizing run_code usage for one setting."""
    stats = summarize_run_code_usage(path)
    if not stats:
        print(f"Insufficient data to plot {title}")
        return

    _fig, axes = plt.subplots(1, 2, figsize=(4, 6))
    # _fig.suptitle(title, fontsize=14, fontweight='bold')

    metrics = [
        ('% Runs', stats['run_code_usage_rate'], 'percent'),
        ('Avg Count / Run', stats['average_run_code_count_per_run'], 'count'),
    ]
    colors = [_tab20[17], _tab20[19]]

    for ax, (subplot_title, value, value_kind), color in zip(axes, metrics, colors):
        ax.bar([0], [value], width=0.6, color=color, edgecolor='white', linewidth=0.8)
        ax.set_title(subplot_title, fontsize=16.5, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels([label], fontsize=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if value_kind == 'percent':
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.text(0, value, f'{value:.0%}', ha='center', va='bottom', fontsize=15)
        else:
            ax.set_ylim(0, 1)
            ax.text(0, value, f'{value:.2f}', ha='center', va='bottom', fontsize=15)
        

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(_fig, output_path, save_pdf=save_pdf)
    plt.close()


def create_run_code_usage_plot_by_outcome(path: str, title: str, output_path: Path, label: str='Setting', save_pdf: bool = False):
    """Create a compact plot summarizing run_code usage split by correct vs failed runs."""
    stats = _summarize_run_code_usage_by_outcome(path)
    if not stats:
        print(f"Insufficient data to plot {title}")
        return

    font_scale = 1.5
    outcomes = ['correct', 'failed']
    outcome_labels = ['Correct', 'Failed']
    x = np.array([0.0, 0.58])
    bar_width = 0.30

    fig, axes = plt.subplots(1, 2, figsize=(4, 6))

    metrics = [
        ('% Runs', 'run_code_usage_rate', 'percent'),
        ('Avg Count / Run', 'average_run_code_count_per_run', 'count'),
    ]
    colors = [_tab20[17], _tab20[19]]

    for ax, (subplot_title, metric_key, value_kind), color in zip(axes, metrics, colors):
        values = [stats[outcome][metric_key] for outcome in outcomes]
        ax.bar(x, values, width=bar_width, color=color, edgecolor='white', linewidth=0.8)
        ax.set_title(subplot_title, fontsize=16.5 * font_scale / 1.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(outcome_labels, fontsize=15 * font_scale / 1.5)
        ax.set_xlim(-0.28, 0.86)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if value_kind == 'percent':
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
            for xpos, value in zip(x, values):
                ax.text(xpos, value, f'{value:.0%}', ha='center', va='bottom', fontsize=15 * font_scale / 1.5)
        else:
            max_value = max(values) if values else 0.0
            ax.set_ylim(0, max(1.0, max_value * 1.2))
            for xpos, value in zip(x, values):
                ax.text(xpos, value, f'{value:.2f}', ha='center', va='bottom', fontsize=15 * font_scale / 1.5)

        ax.set_xlabel(label, fontsize=12 * font_scale / 1.5, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, output_path, save_pdf=save_pdf)
    plt.close()


def summarize_run_code_usage_table(path: str):
    """Summarize agent-full runs by run_code usage and correctness.

    Returns a DataFrame with one row per condition:
    - With run_code
    - Without run_code

    Success rate is the fraction of correct runs as determined by
    evaluate_notebook_correctness on the corresponding patched-executed notebook.
    Avg steps is the mean number of valid tool calls per run.
    """
    path_obj = Path(path)
    rows = {
        'With run_code': {'total_runs': 0, 'correct_runs': 0, 'total_steps': 0},
        'Without run_code': {'total_runs': 0, 'correct_runs': 0, 'total_steps': 0},
    }

    for run_dir in _iter_run_directories(path_obj):
        run_path = Path(run_dir)

        for summary_file in run_path.glob('*_summary.json'):
            instance_id = summary_file.stem.replace('_summary', '')
            notebook_ipynb = run_path / f"{instance_id}_patched_executed.ipynb"
            is_correct, _, _, _ = evaluate_notebook_correctness(notebook_ipynb)

            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                summary = json.load(f)

            total_steps = int(summary.get('statistics', {}).get('total_steps', 0) or 0)
            run_code_count = 0
            for op in summary.get('operations', []):
                action = op.get('action')
                tool_name = extract_tool_name(action)
                if not tool_name:
                    continue

                normalized_tool_name = normalize_tool_name(tool_name)
                if normalized_tool_name == 'run_code':
                    run_code_count += 1

            condition = 'With run_code' if run_code_count > 0 else 'Without run_code'
            bucket = rows[condition]
            bucket['total_runs'] += 1
            bucket['total_steps'] += total_steps
            if is_correct:
                bucket['correct_runs'] += 1

    table_rows = []
    for condition in ('With run_code', 'Without run_code'):
        bucket = rows[condition]
        total_runs = bucket['total_runs']
        table_rows.append({
            'Condition': condition,
            'Success rate': bucket['correct_runs'] / total_runs if total_runs else 0.0,
            'Avg steps': bucket['total_steps'] / total_runs if total_runs else 0.0,
            'Runs': total_runs,
        })

    return pd.DataFrame(table_rows)


def print_run_code_usage_table(path: str):
    """Print the run_code usage comparison table for a single setting."""
    df = summarize_run_code_usage_table(path)
    if df.empty:
        print("No runs found.")
        return df

    display_df = df.copy()
    display_df['Success rate'] = display_df['Success rate'].map(lambda value: f"{value:.2f}")
    display_df['Avg steps'] = display_df['Avg steps'].map(lambda value: f"{value:.1f}")
    display_df = display_df[['Condition', 'Success rate', 'Avg steps']]

    print(display_df.to_string(index=False))
    return df


# def load_manual_validation_outcome(llm_dir: Path):
#     """Load the manual validation outcome string for a model directory if available."""
#     labeled_instances_path = llm_dir / 'analysis' / 'stratified_sampled_instances_labeled.json'
#     if not labeled_instances_path.exists():
#         return None

#     try:
#         with open(labeled_instances_path, 'r', encoding='utf-8') as f:
#             labeled_instances = json.load(f)
#     except (OSError, json.JSONDecodeError) as e:
#         print(f"Warning: failed to load manual validation from {labeled_instances_path}: {e}")
#         return None

#     manual_validation = labeled_instances.get('manual_validation', {})
#     if not isinstance(manual_validation, dict):
#         return None

#     validation_outcome = manual_validation.get('validation_outcome')
#     if isinstance(validation_outcome, str) and validation_outcome.strip():
#         return validation_outcome

#     return None


def format_setting_display_name(setting: str, width: int = 18):
    """Format setting names for compact table display without overflowing cells."""
    return textwrap.fill(setting.replace('_', ' ').title(), width=width)


def get_table_column_widths(columns):
    """Return matplotlib table column widths tuned for the comparison output."""
    widths = {
        'Setting': 0.24,
        'Pass@K (correct)': 0.16,
        'Pass@All (correct)': 0.16,
        'Pass@K (plausible)': 0.16,
        'Pass@All (plausible)': 0.16,
    }
    default_width = 1.0 / max(len(columns), 1)
    return [widths.get(column, default_width) for column in columns]

def main(base_dir: Path):
    """Main function to generate all plots."""
    output_dir = base_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {base_dir}...")
    
    # Aggregate all runs from the base directory
    combined_aggregated = aggregate_directory(base_dir)
    
    if not combined_aggregated:
        print(f"Warning: No data found in {base_dir}")
        return
    
    print(f"  Found {len(combined_aggregated)} steps")
    
    # Create combined plot for all runs
    title = 'Operations by Step - All Runs Combined'
    output_path = output_dir / 'all_runs_operations.png'
    create_stacked_bar_chart(combined_aggregated, title, output_path)
    
    print("\n" + "="*60)
    print("Summary Complete!")
    print(f"Charts saved to: {output_dir.absolute()}")
    print("="*60)


def _normalize_result_instance_name(instance_name):
    """Normalize a result instance reference to the base benchmark instance id."""
    if not instance_name:
        return None

    normalized_instance_name = str(instance_name).strip()
    if '/' in normalized_instance_name:
        normalized_instance_name = normalized_instance_name.rsplit('/', 1)[-1]

    return normalized_instance_name or None


def _iter_setting_llm_summaries(results_dir: Path, settings: list):
    """Yield (setting, llm_name, summary) tuples for all available overall summaries."""
    results_path = Path(results_dir)

    for setting in settings:
        setting_path = results_path / setting

        if not setting_path.exists():
            print(f"Warning: Setting directory not found: {setting_path}")
            continue

        for llm_dir in sorted(setting_path.iterdir()):
            if not llm_dir.is_dir():
                continue

            summary_file = llm_dir / 'overall_summary.json'
            if not summary_file.exists():
                print(f"Warning: overall_summary.json not found in {llm_dir}")
                continue

            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Error loading {summary_file}: {e}")
                continue

            yield setting, llm_dir.name, summary


def _build_comparison_row(setting: str, summary: dict):
    """Build a single row of the setting comparison table from a summary file."""
    stats = summary.get('statistics', {})
    correct_outcome_dist = stats.get('correct_outcome_distribution', {})
    plausible_outcome_dist = stats.get('plausible_outcome_distribution', {})

    return {
        'Setting': SETTING_LABEL_MAP.get(setting, setting),
        'Pass@K (correct)': round(correct_outcome_dist.get('pass_at_k_rate', np.nan), 3),
        'Pass@All (correct)': round(correct_outcome_dist.get('pass_all_k_rate', np.nan), 3),
        'Pass@K (plausible)': round(plausible_outcome_dist.get('pass_at_k_rate', np.nan), 3),
        'Pass@All (plausible)': round(plausible_outcome_dist.get('pass_all_k_rate', np.nan), 3),
    }


def _extract_evaluated_and_correct_instances(summary: dict):
    """Return the evaluated instance ids and correct instance ids from an overall summary."""
    stats = summary.get('statistics', {})
    correct_outcome_dist = stats.get('correct_outcome_distribution', {})

    correct_instances = {
        _normalize_result_instance_name(instance)
        for instance in correct_outcome_dist.get('correct_instances', [])
    }
    incorrect_instances = {
        _normalize_result_instance_name(instance)
        for instance in correct_outcome_dist.get('incorrect_instances', [])
    }

    evaluated_instances = {
        instance for instance in correct_instances | incorrect_instances if instance
    }

    if evaluated_instances:
        return evaluated_instances, {instance for instance in correct_instances if instance}

    plausible_outcome_dist = stats.get('plausible_outcome_distribution', {})
    plausible_instances = {
        _normalize_result_instance_name(instance)
        for instance in plausible_outcome_dist.get('plausible_instances', [])
    }
    failed_instances = {
        _normalize_result_instance_name(instance)
        for instance in plausible_outcome_dist.get('failed_instances', [])
    }

    evaluated_instances = {
        instance for instance in plausible_instances | failed_instances if instance
    }
    return evaluated_instances, {instance for instance in correct_instances if instance}


def _load_benchmark_bug_labels(benchmark_desc_path: Path, bug_category: str):
    """Load benchmark labels for the requested bug category."""
    bug_category = bug_category.lower().strip()
    if bug_category not in BUG_CATEGORY_COLUMN_MAP:
        valid_categories = ', '.join(sorted(BUG_CATEGORY_COLUMN_MAP))
        raise ValueError(f"Unsupported bug_category '{bug_category}'. Expected one of: {valid_categories}")

    label_column, _ = BUG_CATEGORY_COLUMN_MAP[bug_category]
    benchmark_path = Path(benchmark_desc_path)

    benchmark_df = pd.read_excel(benchmark_path)
    required_columns = {'nb_name'} if bug_category == 'library' else {'nb_name', label_column}
    missing_columns = required_columns - set(benchmark_df.columns)
    if missing_columns:
        missing_columns_str = ', '.join(sorted(missing_columns))
        raise KeyError(f"Missing required columns in {benchmark_path}: {missing_columns_str}")

    if bug_category == 'library':
        benchmark_df = benchmark_df[['nb_name']].copy()
        benchmark_df['bug_label'] = benchmark_df['nb_name'].astype(str).str.rsplit('_', n=1).str[0]
    else:
        benchmark_df = benchmark_df[['nb_name', label_column]].copy()
        benchmark_df['bug_label'] = benchmark_df[label_column]

    benchmark_df['instance_id'] = benchmark_df['nb_name'].fillna('').astype(str).str.strip()
    benchmark_df['bug_label'] = benchmark_df['bug_label'].fillna('').astype(str).str.strip()
    benchmark_df = benchmark_df[(benchmark_df['instance_id'] != '') & (benchmark_df['bug_label'] != '')]
    benchmark_df = benchmark_df.drop_duplicates(subset=['instance_id'], keep='first')

    return benchmark_df[['instance_id', 'bug_label']]


def _order_bug_labels(benchmark_labels: pd.DataFrame, bug_categories: list = None):
    """Return bug labels in a stable plotting order."""
    if bug_categories:
        return [str(category) for category in bug_categories]

    if benchmark_labels.empty:
        return []

    counts = benchmark_labels['bug_label'].value_counts()
    return counts.index.tolist()


def _calculate_pass_at_k_by_bug_category(summary: dict, benchmark_labels: pd.DataFrame, bug_categories: list):
    """Calculate pass@k(correct) rates grouped by the requested bug categories."""
    evaluated_instances, correct_instances = _extract_evaluated_and_correct_instances(summary)
    if not evaluated_instances:
        return None

    evaluated_labels = benchmark_labels[benchmark_labels['instance_id'].isin(evaluated_instances)]
    if evaluated_labels.empty:
        return None

    rates = {}
    for bug_category in bug_categories:
        category_instances = set(
            evaluated_labels.loc[evaluated_labels['bug_label'] == bug_category, 'instance_id']
        )
        if not category_instances:
            rates[bug_category] = np.nan
            continue

        correct_count = len(category_instances & correct_instances)
        rates[bug_category] = correct_count / len(category_instances)

    return rates


def _format_bug_category_label(label: str, width: int = 16):
    """Format a bug category label for display on the x-axis."""
    return textwrap.fill(str(label).replace('_', ' '), width=width)


def _plot_grouped_bug_category_bars(rows, bug_categories, bug_category_display_name, title, output_path: Path,
                                   benchmark_labels: pd.DataFrame = None, save_pdf: bool = False):
    """Plot grouped bars for pass@k(correct) across bug categories and settings."""
    if not rows or not bug_categories:
        print(f"Insufficient data to plot {title}")
        return

    num_settings = len(rows)
    num_categories = len(bug_categories)
    
    # Fixed bar width from reference standard (7 categories + 3 settings looks good)
    # For 3 settings: 1/3 = 0.333 capped at 0.25, so bar_width = 0.25
    bar_width = 0.25
    x = np.arange(num_categories)

    # Calculate figure width proportional to number of categories
    # Reference: 7 categories with fig_width≈5 looks good, so scale = 5/7 ≈ 0.71
    # This keeps bars visually identical width across different category counts
    # by ensuring inches-per-data-unit stays constant
    fig_width = max(4, num_categories * 0.71)
    # Scale height based on number of settings
    fig_height = 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    offsets = (np.arange(num_settings) - (num_settings - 1) / 2.0) * bar_width

    for idx, row in enumerate(rows):
        values = [row['rates'].get(category, np.nan) for category in bug_categories]
        color = SETTING_COLORS.get(row['setting_key'], _tab20[idx % len(_tab20)])
        display_label = SETTING_LABEL_MAP.get(row['setting_key'], row['Setting'])
        ax.bar(
            x + offsets[idx],
            values,
            width=bar_width,
            label=display_label,
            color=color,
            edgecolor='white',
            linewidth=0.8,
            alpha=0.95,
        )

    ax.set_xticks(x)
    category_labels = []
    for category in bug_categories:
        label_text = _format_bug_category_label(category)
        label_text = label_text.replace(' ', '\n')
        if benchmark_labels is not None:
            count = (benchmark_labels['bug_label'] == category).sum()
            label_text = f"{label_text}\n({count})"
        category_labels.append(label_text)
    ax.set_xticklabels(category_labels, fontsize=10)
    # ax.set_ylabel('Pass@K (correct) rate', fontsize=10, fontweight='bold')
    # ax.set_xlabel(bug_category_display_name, fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=max(1, min(num_settings, 4)),
        fontsize=8,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    # ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_path, save_pdf=save_pdf)
    plt.close()

def compare_performance_across_settings(results_dir: Path = Path('results'), 
                                       *,
                                       settings: list,
                                       output_dir: Path = None,
                                       save_pdf: bool = False):
    # correct_rate_col = 'Correct Rate\n(CI 90%, MoE 10%)'
    
    if output_dir is None:
        output_dir = results_dir / 'data_analysis'

    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_by_llm = defaultdict(list)
    
    # Iterate through each setting
    for setting, llm_name, summary in _iter_setting_llm_summaries(results_dir, settings):
        row = _build_comparison_row(setting, summary)
        data_by_llm[llm_name].append(row)
        print(f"Loaded: {setting} / {llm_name}")
    
    if not data_by_llm:
        print("No data found to compare")
        return None
    
    # Create and save a table for each LLM
    all_dfs = {}
    
    for llm_name in sorted(data_by_llm.keys()):
        # Create DataFrame for this LLM
        df = pd.DataFrame(data_by_llm[llm_name])
        
        # Sort by setting order
        setting_order = {s: i for i, s in enumerate(settings)}
        df['_setting_order'] = df['Setting'].map(setting_order)
        df = df.sort_values('_setting_order').drop('_setting_order', axis=1)
        df = df.reset_index(drop=True)
        
        all_dfs[llm_name] = df
        
        # Create a formatted table visualization
        _fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Format dataframe for display (round to 4 decimal places)
        df_display = df.copy()
        df_display['Setting'] = df_display['Setting'].apply(format_setting_display_name)
        
        # Create table
        table = ax.table(cellText=df_display.values, colLabels=df_display.columns,
                         cellLoc='center', loc='center',
                         colWidths=get_table_column_widths(list(df_display.columns)),
                         colColours=['#f0f0f0'] * len(df_display.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.8)
        
        # Alternate row colors
        for i in range(len(df_display) + 1):
            for j in range(len(df_display.columns)):
                cell = table[(i, j)]
                cell.get_text().set_wrap(True)
                if i == 0:
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#E8F0F8' if i % 2 == 0 else 'white')
        
        plt.title(f'Performance Comparison: {llm_name}', 
                  fontsize=14, fontweight='bold', pad=20)
        
        table_path = output_dir / f'performance_comparison_{llm_name.replace("/", "_")}.png'
        save_figure(_fig, table_path, save_pdf=save_pdf)
        plt.close()
    
    # Print summary statistics
    for llm_name in sorted(all_dfs.keys()):
        print("\n" + "="*80)
        print(f"Performance Comparison Summary ({llm_name})")
        print("="*80)
        print(all_dfs[llm_name].to_string(index=False))
    print("="*80)
    
    return all_dfs

def compare_pass_at_k_across_bug_types(results_dir: Path = Path('results'),
                                      benchmark_desc_path: Path = Path('JunoBench/benchmark_desc.xlsx'),
                                      *,
                                      settings: list,
                                      bug_category: str = 'library',
                                      bug_categories: list = None,
                                      category_grouping: dict = None,
                                      output_dir: Path = None,
                                      save_pdf: bool = False):
    """Create grouped bar plots for pass@k(correct) across bug categories.

    bug category (x-axis dimension) can be one of BUG_CATEGORY_COLUMN_MAP:
    - library: nb_name from benchmark_desc.xlsx
    - root_cause: label_root_cause
    - crash_type: label_refined_exp_type

    bug_categories (plural, list): Optional list of specific values within that dimension to display, 
    in a specific order. If None, the plot automatically shows all categories found in the data, sorted by frequency

    category_grouping (dict): Optional dict mapping group names to lists of original category names to combine.
        E.g., {'PyTorch': ['torch', 'torchvision'], 'ML Libraries': ['sklearn', 'lightgbm']}
        Categories not in any group will be displayed individually.
        Pass rates are recalculated as (sum of correct instances in group) / (sum of total instances in group).

    Bars are the selected settings, and the y-axis is pass@k(correct), computed
    from the same correct/incorrect instance lists used by
    compare_performance_across_settings.
    """

    bug_category = bug_category.lower().strip()
    if bug_category not in BUG_CATEGORY_COLUMN_MAP:
        valid_categories = ', '.join(sorted(BUG_CATEGORY_COLUMN_MAP))
        raise ValueError(f"Unsupported bug_category '{bug_category}'. Expected one of: {valid_categories}")

    if output_dir is None:
        output_dir = results_dir / 'data_analysis'

    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_labels = _load_benchmark_bug_labels(benchmark_desc_path, bug_category)
    
    # Apply category grouping if provided
    category_rename_map = {}
    if category_grouping:
        for group_name, original_categories in category_grouping.items():
            for original_cat in original_categories:
                category_rename_map[original_cat] = group_name
        # Create a grouped version of benchmark_labels
        benchmark_labels_grouped = benchmark_labels.copy()
        benchmark_labels_grouped['bug_label'] = benchmark_labels_grouped['bug_label'].map(
            lambda x: category_rename_map.get(x, x)
        )
        benchmark_labels = benchmark_labels_grouped
    bug_category_display_name = BUG_CATEGORY_COLUMN_MAP[bug_category][1]

    summaries_by_llm = defaultdict(list)
    evaluated_instances_by_llm = defaultdict(set)

    for setting, llm_name, summary in _iter_setting_llm_summaries(results_dir, settings):
        evaluated_instances, _ = _extract_evaluated_and_correct_instances(summary)
        evaluated_instances_by_llm[llm_name].update(evaluated_instances)
        summaries_by_llm[llm_name].append((setting, summary))

    if not summaries_by_llm:
        print("No data found to compare")
        return None

    all_plots = {}

    for llm_name in sorted(summaries_by_llm.keys()):
        llm_evaluated_instances = evaluated_instances_by_llm.get(llm_name, set())
        llm_labels = benchmark_labels[benchmark_labels['instance_id'].isin(llm_evaluated_instances)].copy()

        if llm_labels.empty:
            print(f"Insufficient benchmark labels to plot {llm_name}")
            continue

        categories = _order_bug_labels(llm_labels, bug_categories=bug_categories)
        if not categories:
            print(f"No bug categories found for {llm_name}")
            continue

        rows = []
        for setting, summary in summaries_by_llm[llm_name]:
            rates = _calculate_pass_at_k_by_bug_category(summary, llm_labels, categories)
            if rates is None:
                continue

            rows.append({
                'setting_key': setting,
                'Setting': SETTING_LABEL_MAP.get(setting, setting),
                'rates': rates,
            })

        if not rows:
            print(f"Insufficient data to plot {llm_name}")
            continue

        all_plots[llm_name] = rows

        title = f'Pass@K (Correct) by {bug_category_display_name}: {llm_name}'
        output_path = output_dir / f'pass_at_k_correct_by_{bug_category}_{llm_name.replace("/", "_")}.png'
        _plot_grouped_bug_category_bars(rows, categories, bug_category_display_name, title, output_path,
                                       benchmark_labels=llm_labels, save_pdf=save_pdf)

    return all_plots
