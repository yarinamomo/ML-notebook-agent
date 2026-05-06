"""
Visualize the patterns in run_code operations.

This script automatically runs the analysis first to ensure data is up to date,
then generates visualizations from the analyzed data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
# Import the analyze module to run analysis first
from data_analysis import run_code_analyze

def plot_category_distribution(input_dir, output_dir):
    """Plot the distribution of run_code operation categories."""
    
    # Load the JSON data
    json_file = input_dir / 'run_code_operations.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    operations = data['operations']
    
    # Count categories
    category_counts = Counter()
    
    for op in operations:
        for cat in op['categories']:
            category_counts[cat] += 1
    
    # Prepare data for plotting
    categories = [] # 4 main categories: prints, structure, type, value
    counts = []
    percentages = []
    
    total_ops = len(operations)
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        categories.append(cat.replace('_', ' ').title())
        counts.append(count)
        percentages.append((count / total_ops) * 100)
    
    # Scale fonts by 1.5x for better readability
    scale = 1.4
    label_font = int(10 * scale)
    title_font = int(14 * scale)
    value_font = int(9 * scale)

    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4))

    # Plot: Vertical bar chart
    # Color strategy for presentation: gray for the top category,
    # and progressively lighter blue shades for the remaining categories.
    colors = [plt.cm.Greys(0.4)]
    shade_values = np.linspace(0.75, 0.5, len(categories) - 1)
    colors.extend([plt.cm.Blues(v) for v in shade_values])

    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)

    # Small legend for the two conceptual groups.
    first_color = colors[0]
    runinfo_colors = list(colors[1:4])

    runinfo_handle = tuple(Patch(facecolor=c, edgecolor='black') for c in runinfo_colors[:3])
    legend_handles = [
        Patch(facecolor=first_color, edgecolor='black'),
        runinfo_handle,
    ]
    legend_labels = ['Overall prints', 'Runinfo categories']
    ax1.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='upper right',
        fontsize=max(8, int(value_font * 0.85)),
        frameon=True,
    )

    # Add value labels above each bar
    y_offset = max(counts) * 0.01 if max(counts) > 0 else 0.1
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + y_offset,
                 f'{count}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=value_font)

    ax1.set_ylabel('Number of Operations', fontsize=label_font, fontweight='bold')
    # ax1.set_title('Tool "run_code" Categories Distribution', fontsize=title_font, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(counts) * 1.15)
    ax1.set_xticklabels([c for c in categories], rotation=45, ha='right', fontsize=value_font)
    ax1.tick_params(axis='y', labelsize=value_font)

    plt.tight_layout()

    # Save figures: PNG and PDF
    output_file_png = output_dir / 'run_code_categories_visualization.png'
    output_file_pdf = output_dir / 'run_code_categories_visualization.pdf'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Visualizations saved to: {output_file_png} and {output_file_pdf}")
    plt.close()

def plot_code_length_distribution(input_dir, output_dir):
    """Plot the distribution of code lengths."""
    
    # Load the JSON data
    json_file = input_dir / 'run_code_operations.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    operations = data['operations']
    
    # Get code lengths (lines of code)
    code_loc = [len(op['code'].splitlines()) for op in operations]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram
    ax1.hist(code_loc, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.median(code_loc), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(code_loc):.1f}')
    ax1.axvline(np.mean(code_loc), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(code_loc):.1f}')
    
    ax1.set_xlabel('Code lines', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of run_code Code lines', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot
    box = ax2.boxplot([code_loc], vert=True, patch_artist=True, 
                       labels=['All Operations'])
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][0].set_edgecolor('black')
    
    # Add statistics text
    stats_text = f"Min: {min(code_loc)}\n"
    stats_text += f"Q1: {np.percentile(code_loc, 25):.1f}\n"
    stats_text += f"Median: {np.median(code_loc):.1f}\n"
    stats_text += f"Q3: {np.percentile(code_loc, 75):.1f}\n"
    stats_text += f"Max: {max(code_loc)}"
    
    ax2.text(1.15, np.median(code_loc), stats_text, 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_ylabel('Code Length (lines)', fontsize=12, fontweight='bold')
    ax2.set_title('run_code Code Length Statistics', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'run_code_length_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()


def plot_operations_by_run(input_dir, output_dir):
    """Plot comparison of operations across runs."""
    
    # Load the JSON data
    json_file = input_dir / 'run_code_operations.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    operations = data['operations']
    
    # Separate by run
    run_data = {1: [], 2: [], 3: []}
    
    for op in operations:
        run_data[op['run']].append(op)
    
    # Count categories per run    
    categories_per_run = {}
    for run_num, ops in run_data.items():
        category_counts = Counter()
        for op in ops:
            for cat in op['categories']:
                category_counts[cat] += 1
        categories_per_run[run_num] = category_counts
    
    # Get all unique categories
    all_categories = set()
    for counts in categories_per_run.values():
        all_categories.update(counts.keys())
    
    all_categories = sorted(all_categories)
    
    # Prepare data for grouped bar chart
    x = np.arange(len(all_categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    for i, (run_num, counts) in enumerate(sorted(categories_per_run.items())):
        values = [counts.get(cat, 0) for cat in all_categories]
        ax.bar(x + i*width, values, width, label=f'Run {run_num}', 
               color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Operations', fontsize=12, fontweight='bold')
    ax.set_title('run_code Operations by Category and Run', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in all_categories], 
                        rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'run_code_by_run_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()


def plot_pie_chart_distributions(input_dir, output_dir):
    """Plot category co-occurrence heatmap."""
    
    # Load the JSON data
    json_file = input_dir / 'run_code_operations.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    operations = data['operations']

    # Plot: Top libraries pie chart
    # Extract top methods
    method_counts = Counter()
    for op in operations:
        for method in op['methods']:
            method_counts[method] += 1
    
    top_methods = dict(method_counts.most_common(10))
    others_count = sum(count for method, count in method_counts.items() if method not in top_methods)
    
    if others_count > 0:
        top_methods['Others'] = others_count
    
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create pie chart
    colors2 = plt.cm.Set3(np.linspace(0, 1, len(top_methods)))
    wedges, texts, autotexts = ax.pie(
        top_methods.values(), 
        labels=top_methods.keys(),
        autopct='%1.1f%%',
        colors=colors2,
        startangle=90,
        textprops={'fontsize': 9}
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title('Top Methods Used in run_code', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'run_code_methods_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()


def main(base_dir: Path):
    """Generate all visualizations."""
    input_dir = base_dir / 'analysis'
    output_dir = base_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("STEP 1: Running analysis to ensure data is up to date...")
    print("="*60)
    
    # Run the analysis first to generate/update the JSON data
    run_code_analyze.main(input_dir)
    
    print("\n" + "="*60)
    print("STEP 2: Generating visualizations...")
    print("="*60)
    
    plot_category_distribution(input_dir, output_dir)
    plot_code_length_distribution(input_dir, output_dir)
    plot_operations_by_run(input_dir, output_dir)
    plot_pie_chart_distributions(input_dir, output_dir)
    
    print("\nAll visualizations generated successfully!")
    print(f"Charts saved to: {output_dir.absolute()}")
