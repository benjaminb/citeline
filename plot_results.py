import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import glob
import json
import os
from pprint import pprint
import sys

"""
Provide this program a directory, and it will search for all .json files (assumed to be experiment results)
and plot them together

USAGE:
python plot_results.py --dir <directory> --k <value of k> --outfile <output filename>
"""


def get_json_files(path):
    return glob.glob(os.path.join(path, "**/*.json"), recursive=True)


def path_to_label(path):
    # Convert a file path to a label by removing the base directory and file extension
    filename = os.path.basename(path)
    return filename.replace(".json", "")


def make_label_to_data_dict(json_files: list[str], metric: str = "hitrate") -> dict:
    data = dict()
    metric_key = f"average_{metric}_at_k"
    for file in json_files:
        results = json.load(open(file))
        values = results.get(metric_key, results["average_stats"].get(metric_key, None))
        label = results["config"].get("plot_label", path_to_label(file))
        data[label] = values
    return data


def plot_results(data: dict, path: str, k: int = 1000, name: str = None, title: str = None):
    # Define color palette
    colors = plt.get_cmap("tab20").colors  # or 'tab20'
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    k_values = range(1, k + 1)
    plt.figure(figsize=(16, 16))

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.6)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    lines = []
    # place labels near k=200 (or at k if k < 200)
    label_x = min(200, k)
    min_gap = 0.02  # minimum vertical gap between labels

    items = []
    for label, rates_at_k in sorted(data.items()):
        rates_at_k_trunc = rates_at_k[:k]
        y = rates_at_k_trunc[10]  # value at k=100
        items.append({"label": label, "hitrates": rates_at_k_trunc, "y": float(y)})

    # sort by y descending so high values labeled from top of the label band
    items.sort(key=lambda x: x["y"], reverse=True)

    # Place labels in a bottom band (e.g., from base_y up to max_label_y).
    # Compute spacing dynamically so we fit all labels in the band.
    n = len(items)
    base_y = 0.02
    max_label_y = 0.20
    if n <= 1:
        label_ys = [base_y]
    else:
        available = max_label_y - base_y
        spacing = available / (n - 1)
        # If spacing would be larger than min_gap, clamp to min_gap to avoid very sparse labels;
        # otherwise use computed spacing (this may compress if many labels).
        if spacing > min_gap:
            spacing = min_gap
            # recompute top of band to fit with min_gap
            top_needed = base_y + spacing * (n - 1)
            if top_needed > max_label_y:
                # compress into available if needed
                spacing = available / (n - 1)
        # Assign label positions so the highest-scoring item gets the highest position in the band
        label_ys = [base_y + (n - 1 - i) * spacing for i in range(n)]

    # plot lines and place labels at computed bottom-band positions
    for it, y_label in zip(items, label_ys):
        label = it["label"]
        rates_at_k_trunc = it["hitrates"]
        (line,) = plt.plot(
            k_values, rates_at_k_trunc, drawstyle="steps-post", linestyle="-", lw=4.0, alpha=1.0, label=label
        )
        lines.append((line, label, rates_at_k_trunc))

    # Add annotations for highest scoring line at k=50 and at k=100, 200, 300, etc.
    k_points = []
    if k >= 50:
        k_points.append(50)
    k_points.extend(range(100, k + 1, 100))

    # Annotate the highest scoring line at this k value
    # for k_val in k_points:
    #     k_idx = k_val - 1  # Convert to 0-indexed

    #     # Find the line with the highest hitrate at this k value
    #     max_hitrate = -1
    #     best_line = None
    #     best_color = None

    #     for line, label, hitrates_trunc in lines:
    #         if k_idx < len(hitrates_trunc):
    #             hitrate_at_k = hitrates_trunc[k_idx]
    #             if hitrate_at_k > max_hitrate:
    #                 max_hitrate = hitrate_at_k
    #                 best_line = label
    #                 best_color = line.get_color()

    #     if best_line is not None and max_hitrate >= 0:
    #         plt.annotate(
    #             f"{max_hitrate:.3f}",
    #             xy=(k_val, max_hitrate),
    #             xytext=(0, 25),  # 25 points above
    #             textcoords="offset points",
    #             fontsize=14,
    #             color=best_color,
    #             ha="center",
    #             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
    #         )

    plt.xlabel("Top-k", fontsize=30)
    plt.ylabel("Score", fontsize=30)
    plot_title = title if title else "HitRate@k"
    plt.title(plot_title, fontsize=30)

    # Build legend sorted by score at k=50 (descending)
    sorted_lines = sorted(lines, key=lambda t: t[2][49], reverse=True)
    handles = [t[0] for t in sorted_lines]
    labels = [t[1] for t in sorted_lines]
    # custom_text = plt.Line2D([0], [0], color="none", label="Ranking@k=50")
    # handles.insert(0, custom_text)
    # labels.insert(0, "Ranking@k=50")
    plt.legend(handles=handles, labels=labels, fontsize=30, loc="lower right", framealpha=0.9)

    # Control grid lines
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(25))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.7, color="gray", alpha=0.5)

    # Tick label size
    ax.tick_params(axis="both", which="major", labelsize=30)

    # Manually add x-axis tick label at first position, which is k=25
    current_ticks = list(ax.get_xticks())
    if 25 not in current_ticks:
        current_ticks.insert(0, 25)
        ax.set_xticks(sorted(current_ticks))

    # ensure full k range is visible (start x-axis at 25)
    plt.xlim(25, k)
    # plt.ylim(0.6, 1.0)

    # basename = path.split("/")[0]
    outfile = name if name else f"plot_results.png"
    save_path = os.path.join(path, outfile)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--dir", type=str, help="Directory containing experiment results JSON files")
    parser.add_argument("--k", type=int, default=200, help="Value of k for top-k plotting")
    parser.add_argument("--outfile", type=str, help="Output filename for the plot image")
    parser.add_argument("--title", type=str, help="Title for the plot (optional)")
    parser.add_argument(
        "--metric",
        type=str,
        default="hitrate",
        choices=["hitrate", "recall"],
        help="Metric to plot: 'hitrate' or 'recall' (default: hitrate)",
    )
    args = parser.parse_args()

    directory = args.dir
    k = args.k
    outfile = args.outfile
    title = args.title
    metric = args.metric

    # Set default title based on metric if not provided
    if not title:
        title = f"{metric.capitalize()}@k"

    # Establish what files will be included
    # path = os.path.join(BASE_DIR, directory)
    json_files = get_json_files(directory)
    print(f"Found {len(json_files)} JSON files in {directory}")
    data = make_label_to_data_dict(json_files, metric=metric)

    plot_results(data, path=directory, k=k, name=outfile, title=title)


if __name__ == "__main__":
    main()
