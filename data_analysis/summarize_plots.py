"""
Summarize and visualize operations taken by LLM runs across different test cases.

This script generates stacked bar charts showing the count of different operations/actions
taken by the LLM agent at each step of execution.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
    "Submitted"
]


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
            
        except Exception as e:
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
    
    for instance_id, operations_by_step in data_dict.items():
        for step, tool_counts in operations_by_step.items():
            for tool_name, count in tool_counts.items():
                aggregated[step][tool_name] += count
    
    return aggregated


def create_stacked_bar_chart(aggregated_data, title, output_path):
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
    
    # Get all steps and sort them
    steps = sorted(aggregated_data.keys())
    
    # Get all tool names that appear in the data
    all_tools = set()
    for tool_counts in aggregated_data.values():
        all_tools.update(tool_counts.keys())
    
    # Sort tools using VALID_TOOLS order
    tools = [tool for tool in VALID_TOOLS if tool in all_tools]
    # Add any tools not in the predefined order at the end
    tools.extend(sorted([tool for tool in all_tools if tool not in VALID_TOOLS]))
    
    # Prepare data for stacking
    data_matrix = np.zeros((len(tools), len(steps)))
    
    for i, step in enumerate(steps):
        for j, tool in enumerate(tools):
            data_matrix[j, i] = aggregated_data[step].get(tool, 0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(12, len(steps) * 0.5), 8))
    
    # Define colors for each tool
    colors = plt.cm.tab10(np.linspace(0, 1, len(tools)))
    
    # Create stacked bars
    bottom = np.zeros(len(steps))
    bars = []
    
    for j, tool in enumerate(tools):
        bar = ax.bar(range(len(steps)), data_matrix[j], bottom=bottom, 
                     label=tool, color=colors[j], edgecolor='white', linewidth=0.5)
        bars.append(bar)
        bottom += data_matrix[j]
    
    # Customize the plot
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operation Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45 if len(steps) > 20 else 0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart to {output_path}")
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


def create_comparison_chart(path1: str, path2: str, title: str, output_path: Path, label1: str='Setting 1', label2: str='Setting 2', mode: str='diff'):
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
        colors = plt.cm.tab10(np.linspace(0, 1, len(tools)))
        
        # Plot difference lines for each tool
        for j, tool in enumerate(tools):
            steps_list = list(all_steps)
            data1 = np.array([aggregated_data1.get(step, {}).get(tool, 0) for step in steps_list])
            data2 = np.array([aggregated_data2.get(step, {}).get(tool, 0) for step in steps_list])
            
            # Calculate difference (positive = more in setting 1, negative = more in setting 2)
            differences = data1 - data2
            
            # Plot line for this tool's difference
            ax.plot(range(len(steps_list)), differences, marker='o', linestyle='-', linewidth=2.5,
                   color=colors[j], label=tool, alpha=0.8, markersize=6)
        
        # Add a horizontal line at y=0 to show the neutral point
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Customize the plot
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Difference', fontsize=12, fontweight='bold')
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Append mode to filename before extension (e.g., comparison_diff.png)
    final_path = output_path.parent / (output_path.stem + f'_{mode}' + output_path.suffix)
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart to {final_path}")
    plt.close()

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
