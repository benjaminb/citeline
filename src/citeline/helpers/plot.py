import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


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
