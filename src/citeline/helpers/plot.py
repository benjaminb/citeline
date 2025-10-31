import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm
from pathlib import Path
import json
from collections import defaultdict


def _extract_prefix(filename: str, min_prefix_length: int = 10) -> str:
    """
    Extracts common prefix from filename by finding the part before numbers/timestamps.

    Args:
        filename: The filename to extract prefix from
        min_prefix_length: Minimum length for a valid prefix

    Returns:
        The extracted prefix
    """
    # Remove extension
    name = Path(filename).stem

    # Find where the pattern changes (e.g., before timestamps or version numbers)
    # Look for common separators followed by numbers/dates
    import re

    # Try to find a pattern like _n12345_ or _20251017_ or similar
    match = re.search(r"(_n\d+_|_\d{8}_)", name)
    if match:
        prefix = name[: match.start()]
        if len(prefix) >= min_prefix_length:
            return prefix

    # Fallback: use the whole name
    return name


def _group_files_by_prefix(files: list[Path]) -> dict[str, list[Path]]:
    """
    Groups files by their common prefix.

    Args:
        files: List of file paths

    Returns:
        Dictionary mapping prefix to list of files with that prefix
    """
    groups = defaultdict(list)

    for file in files:
        prefix = _extract_prefix(file.name)
        groups[prefix].append(file)

    return dict(groups)


def _assign_colors_to_groups(groups: dict[str, list[Path]], colormap: str = "tab10") -> dict[Path, str]:
    """
    Assigns colors from a colormap to files based on their groups.

    Args:
        groups: Dictionary mapping prefix to list of files
        colormap: Name of matplotlib colormap to use

    Returns:
        Dictionary mapping file path to color
    """
    file_colors = {}
    cmap = cm.get_cmap(colormap)

    # Sort groups for consistent ordering
    sorted_groups = sorted(groups.items())

    for group_idx, (prefix, files) in enumerate(sorted_groups):
        if len(sorted_groups) > 1:
            # Use different colors for different groups
            color = cmap(group_idx / max(1, len(sorted_groups) - 1))
        else:
            # Single group - use first color
            color = cmap(0)

        for file in files:
            file_colors[file] = color

    return file_colors


def plot_multiple_results_from_dir(
    results_dir: str,
    output_path: str,
    metric: str = "hitrate",
    k_cutoff: int = None,
    pattern: str = "*.json",
    colormap: str = "tab10",
    title: str = None,
):
    """
    Plots a metric from multiple result files in a directory.
    Files with the same prefix are assigned colors from a progressive colormap.

    Args:
        results_dir: Directory containing result JSON files
        output_path: Path where the plot will be saved
        metric: Which metric to plot ("hitrate", "iou", or "recall")
        k_cutoff: Maximum k value to plot. If None, plots all values
        pattern: Glob pattern to match result files
        colormap: Matplotlib colormap name for coloring groups
        title: Title for the plot. If None, auto-generated
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' found in {results_dir}")
        return

    # Group files by prefix and assign colors
    groups = _group_files_by_prefix(files)
    file_colors = _assign_colors_to_groups(groups, colormap)

    # Create figure
    plt.figure(figsize=(14, 8))

    # Plot each file
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        # Extract the metric data
        metric_key = f"average_{metric}_at_k"
        if metric_key not in data:
            print(f"Warning: {metric_key} not found in {file.name}")
            continue

        metric_data = np.array(data[metric_key])

        if k_cutoff is not None:
            metric_data = metric_data[:k_cutoff]

        k_values = list(range(1, len(metric_data) + 1))

        # Get label from config if available
        label = data.get("config", {}).get("plot_label", file.stem)

        # Plot with assigned color
        plt.plot(k_values, metric_data, linestyle="-", label=label, color=file_colors[file], alpha=0.8, linewidth=1.5)

    plt.xlabel("Top-k")
    plt.ylabel(metric.capitalize())
    plt.title(title or f"{metric.capitalize()}@k Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add grid
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")
    print(f"Plotted {len(files)} files from {len(groups)} groups")


def plot_stats_at_k(
    stats: dict[str, np.ndarray],
    output_path: str,
    k_cutoff: int = None,
    annotation_interval: int = 50,
    title: str = "Stats@k",
):
    """
    Plots hitrate, IoU, and recall metrics at different k values.

    Args:
        stats: Dictionary with keys "hitrate", "iou", "recall" and values as numpy arrays
               (output from compute_averages in statistics.py)
        output_path: Path where the plot will be saved (e.g., "results/plot.png")
        k_cutoff: Maximum k value to plot. If None, plots all values. Default: None
        annotation_interval: Interval for adding value annotations. Default: 50
        title: Title for the plot. Default: "Stats@k"
    """
    # Get the arrays and apply k_cutoff if specified
    hitrate_at_k = stats["hitrate"]
    iou_at_k = stats["iou"]
    recall_at_k = stats["recall"]

    if k_cutoff is not None:
        hitrate_at_k = hitrate_at_k[:k_cutoff]
        iou_at_k = iou_at_k[:k_cutoff]
        recall_at_k = recall_at_k[:k_cutoff]

    top_k = len(hitrate_at_k)
    k_values = list(range(1, top_k + 1))

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot the lines
    plt.plot(k_values, hitrate_at_k, linestyle="-", label="Average Hit Rate@k", color="blue")
    plt.plot(k_values, iou_at_k, linestyle="-", label="Average IoU@k", color="green")
    plt.plot(k_values, recall_at_k, linestyle="-", label="Average Recall@k", color="red")

    # Add marker and label for maximal IoU point
    best_k_for_iou = int(np.argmax(iou_at_k)) + 1  # +1 for 1-indexed k
    max_iou_value = iou_at_k[best_k_for_iou - 1]
    plt.scatter(best_k_for_iou, max_iou_value, color="green", s=100, zorder=5, marker="o")
    plt.annotate(
        f"Max IoU: {max_iou_value:.3f} at k={best_k_for_iou}",
        xy=(best_k_for_iou, max_iou_value),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=10,
        color="green",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
    )

    # Add annotations at regular intervals
    for i in range(0, len(k_values), annotation_interval):
        k = k_values[i]

        # Annotate hit rate
        plt.annotate(
            f"{hitrate_at_k[i]:.3f}",
            xy=(k, hitrate_at_k[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="blue",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

        # Annotate IoU
        plt.annotate(
            f"{iou_at_k[i]:.3f}",
            xy=(k, iou_at_k[i]),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=8,
            color="green",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

        # Annotate recall
        plt.annotate(
            f"{recall_at_k[i]:.3f}",
            xy=(k, recall_at_k[i]),
            xytext=(5, -25),
            textcoords="offset points",
            fontsize=8,
            color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    plt.xlabel("Top-k")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()

    # Add grid lines at 0.05 intervals but labels every 0.1
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Labels every 0.1
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))  # Grid lines every 0.05
    plt.grid(True, alpha=0.3, which="both")  # Show both major and minor grid lines

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")
