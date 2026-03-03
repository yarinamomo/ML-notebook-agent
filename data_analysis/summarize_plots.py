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


def main():
    """Main function to generate all plots."""
    base_dir = Path('trajectories_monday/glm-4.7-355b')
    output_dir = base_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Process each run individually
    run_data = {}
    
    for run_num in [1, 2, 3]:
        run_dir = base_dir / f'run_{run_num}'
        
        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {run_dir}...")
        
        # Load data for this run
        data = load_summary_data(run_dir)
        print(f"  Loaded {len(data)} test cases")
        
        # Aggregate operations for this run
        aggregated = aggregate_operations(data)
        print(f"  Found {len(aggregated)} steps")
        
        # Store for later aggregation
        run_data[run_num] = aggregated
        
        # Create individual run plot
        title = f'Operations by Step - Run {run_num}'
        output_path = output_dir / f'run_{run_num}_operations.png'
        create_stacked_bar_chart(aggregated, title, output_path)
    
    # Create combined plot for all runs
    print("\nCreating combined plot for all runs...")
    
    # Aggregate all runs together
    combined_aggregated = defaultdict(lambda: defaultdict(int))
    
    for run_num, aggregated in run_data.items():
        for step, tool_counts in aggregated.items():
            for tool_name, count in tool_counts.items():
                combined_aggregated[step][tool_name] += count
    
    title = 'Operations by Step - All Runs Combined'
    output_path = output_dir / 'all_runs_operations.png'
    create_stacked_bar_chart(combined_aggregated, title, output_path)
    
    print("\n" + "="*60)
    print("Summary Complete!")
    print(f"Charts saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == '__main__':
    main()
