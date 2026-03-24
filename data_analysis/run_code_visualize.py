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

# Import the analyze module to run analysis first
from data_analysis import run_code_analyze


def plot_category_distribution(input_dir, output_dir):
    """Plot the distribution of run_code operation categories."""

    # Load the JSON data
    json_file = input_dir / "run_code_operations.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    operations = data["operations"]

    # Count categories
    category_counts = Counter()

    for op in operations:
        for cat in op["categories"]:
            category_counts[cat] += 1

    # Prepare data for plotting
    categories = []
    counts = []
    percentages = []

    total_ops = len(operations)

    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        categories.append(cat.replace("_", " ").title())
        counts.append(count)
        percentages.append((count / total_ops) * 100)

    # Create figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[0.6, 0.4])
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

    # Plot 1: Horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
    bars = ax1.barh(categories, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, count, pct in zip(bars, counts, percentages):
        width = bar.get_width()
        ax1.text(
            width + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax1.set_xlabel("Number of Operations", fontsize=12, fontweight="bold")
    ax1.set_title(
        'Tool "run_code" Categories Distribution', fontsize=14, fontweight="bold"
    )
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    ax1.set_xlim(0, max(counts) * 1.15)

    # # Plot 2: Co-occurrence heatmap of pair overlap categories
    # # Check if we have overlap data
    # if not isinstance(data, dict) or 'overlaps' not in data:
    #     print("No overlap data found. Run analysis first.")
    #     return

    # overlaps = data['overlaps']

    # # Remove 'other' and 'print_output' from categories list
    # categories = [cat for cat in overlaps['all_categories'] if cat != 'other' and cat != 'print_output']
    # n_cats = len(categories)

    # # Build co-occurrence matrix
    # cooccur_matrix = np.zeros((n_cats, n_cats))

    # # Diagonal: count of each category
    # category_counts = Counter()
    # for op in operations:
    #     for cat in op['categories']:
    #         category_counts[cat] += 1

    # for i, cat in enumerate(categories):
    #     cooccur_matrix[i, i] = category_counts[cat]

    # # Off-diagonal: co-occurrence counts
    # for pair_str, count in overlaps['co_occurrence_pairs'].items():
    #     cat1, cat2 = pair_str.split('|')
    #     # Skip pairs involving 'other' or 'print_output' categories
    #     if cat1 == 'other' or cat2 == 'other' or cat1 == 'print_output' or cat2 == 'print_output':
    #         continue
    #     i = categories.index(cat1)
    #     j = categories.index(cat2)
    #     cooccur_matrix[i, j] = count
    #     cooccur_matrix[j, i] = count  # Symmetric

    # # Plot heatmap
    # im = ax2.imshow(cooccur_matrix, cmap='YlOrRd', aspect='auto')

    # # Set ticks and labels
    # cat_labels = [cat.replace('_', ' ').title() for cat in categories]
    # ax2.set_xticks(np.arange(n_cats))
    # ax2.set_yticks(np.arange(n_cats))
    # ax2.set_xticklabels(cat_labels, rotation=45, ha='right')
    # ax2.set_yticklabels(cat_labels)

    # # Add text annotations
    # for i in range(n_cats):
    #     for j in range(n_cats):
    #         count = int(cooccur_matrix[i, j])
    #         if count > 0:
    #             color = 'white' if cooccur_matrix[i, j] > cooccur_matrix.max() / 2 else 'black'
    #             ax2.text(j, i, str(count), ha='center', va='center',
    #                     color=color, fontsize=9, fontweight='bold')

    # ax2.set_title('Runinfo Category Co-occurrence Matrix\n(diagonal = total count, off-diagonal = overlap count)',
    #               fontsize=12, fontweight='bold')

    # # Add colorbar
    # cbar = plt.colorbar(im, ax=ax2)
    # cbar.set_label('Number of Operations', rotation=270, labelpad=20)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "run_code_categories_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")
    plt.close()


def plot_code_length_distribution(input_dir, output_dir):
    """Plot the distribution of code lengths."""

    # Load the JSON data
    json_file = input_dir / "run_code_operations.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    operations = data["operations"]

    # Get code lengths (lines of code)
    code_loc = [len(op["code"].splitlines()) for op in operations]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram
    ax1.hist(code_loc, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.median(code_loc),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(code_loc):.1f}",
    )
    ax1.axvline(
        np.mean(code_loc),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(code_loc):.1f}",
    )

    ax1.set_xlabel("Code lines", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title("Distribution of run_code Code lines", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Box plot
    box = ax2.boxplot(
        [code_loc], vert=True, patch_artist=True, labels=["All Operations"]
    )
    box["boxes"][0].set_facecolor("lightblue")
    box["boxes"][0].set_edgecolor("black")

    # Add statistics text
    stats_text = f"Min: {min(code_loc)}\n"
    stats_text += f"Q1: {np.percentile(code_loc, 25):.1f}\n"
    stats_text += f"Median: {np.median(code_loc):.1f}\n"
    stats_text += f"Q3: {np.percentile(code_loc, 75):.1f}\n"
    stats_text += f"Max: {max(code_loc)}"

    ax2.text(
        1.15,
        np.median(code_loc),
        stats_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2.set_ylabel("Code Length (lines)", fontsize=12, fontweight="bold")
    ax2.set_title("run_code Code Length Statistics", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "run_code_length_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")
    plt.close()


def plot_operations_by_run(input_dir, output_dir):
    """Plot comparison of operations across runs."""

    # Load the JSON data
    json_file = input_dir / "run_code_operations.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    operations = data["operations"]

    # Separate by run
    run_data = {1: [], 2: [], 3: []}

    for op in operations:
        run_data[op["run"]].append(op)

    # Count categories per run
    categories_per_run = {}
    for run_num, ops in run_data.items():
        category_counts = Counter()
        for op in ops:
            for cat in op["categories"]:
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

    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

    for i, (run_num, counts) in enumerate(sorted(categories_per_run.items())):
        values = [counts.get(cat, 0) for cat in all_categories]
        ax.bar(
            x + i * width,
            values,
            width,
            label=f"Run {run_num}",
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Operations", fontsize=12, fontweight="bold")
    ax.set_title(
        "run_code Operations by Category and Run", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [cat.replace("_", " ").title() for cat in all_categories],
        rotation=45,
        ha="right",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "run_code_by_run_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")
    plt.close()


def plot_pie_chart_distributions(input_dir, output_dir):
    """Plot category co-occurrence heatmap."""

    # Load the JSON data
    json_file = input_dir / "run_code_operations.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    operations = data["operations"]

    # Plot: Top libraries pie chart
    # Extract top methods
    method_counts = Counter()
    for op in operations:
        for method in op["methods"]:
            method_counts[method] += 1

    top_methods = dict(method_counts.most_common(10))
    others_count = sum(
        count for method, count in method_counts.items() if method not in top_methods
    )

    if others_count > 0:
        top_methods["Others"] = others_count

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create pie chart
    colors2 = plt.cm.Set3(np.linspace(0, 1, len(top_methods)))
    wedges, texts, autotexts = ax.pie(
        top_methods.values(),
        labels=top_methods.keys(),
        autopct="%1.1f%%",
        colors=colors2,
        startangle=90,
        textprops={"fontsize": 9},
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")

    ax.set_title("Top Methods Used in run_code", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "run_code_methods_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")
    plt.close()


def main(base_dir: Path):
    """Generate all visualizations."""
    input_dir = base_dir / "analysis"
    output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Running analysis to ensure data is up to date...")
    print("=" * 60)

    # Run the analysis first to generate/update the JSON data
    run_code_analyze.main(input_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Generating visualizations...")
    print("=" * 60)

    plot_category_distribution(input_dir, output_dir)
    plot_code_length_distribution(input_dir, output_dir)
    plot_operations_by_run(input_dir, output_dir)
    plot_pie_chart_distributions(input_dir, output_dir)

    print("\nAll visualizations generated successfully!")
    print(f"Charts saved to: {output_dir.absolute()}")
