import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
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


def make_label_to_data_dict(json_files: list[str]) -> dict:
    data = dict()
    for file in json_files:
        results = json.load(open(file))
        hitrates = results["average_hitrate_at_k"]
        label = results["config"].get("plot_label", path_to_label(file))
        data[label] = hitrates
    return data


def plot_results(data: dict, path: str, k: int = 1000, name: str = None):
    k_values = range(1, k + 1)
    plt.figure(figsize=(16, 16))

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.6)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    lines = []
    # place labels near k=200 (or at k if k < 200)
    label_x = min(200, k)
    min_gap = 0.02  # minimum vertical gap between labels

    items = []
    for label, hitrates in sorted(data.items()):
        hitrates_trunc = hitrates[:k]
        y = hitrates_trunc[10]  # value at k=100
        items.append({"label": label, "hitrates": hitrates_trunc, "y": float(y)})

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
        hitrates_trunc = it["hitrates"]
        (line,) = plt.plot(
            k_values, hitrates_trunc, drawstyle="steps-post", linestyle="-", lw=2.5, alpha=0.6, label=label
        )
        lines.append((line, label, hitrates_trunc))

    # Add annotations for highest scoring line at k=50 and at k=100, 200, 300, etc.
    k_points = []
    if k >= 50:
        k_points.append(50)
    k_points.extend(range(100, k + 1, 100))

    for k_val in k_points:
        k_idx = k_val - 1  # Convert to 0-indexed
        # Find the line with the highest hitrate at this k value
        max_hitrate = -1
        best_line = None
        best_color = None

        for line, label, hitrates_trunc in lines:
            if k_idx < len(hitrates_trunc):
                hitrate_at_k = hitrates_trunc[k_idx]
                if hitrate_at_k > max_hitrate:
                    max_hitrate = hitrate_at_k
                    best_line = label
                    best_color = line.get_color()

        # Annotate the highest scoring line at this k value
        if best_line is not None and max_hitrate >= 0:
            plt.annotate(
                f"{max_hitrate:.3f}",
                xy=(k_val, max_hitrate),
                xytext=(0, 25),  # 25 points above
                textcoords="offset points",
                fontsize=14,
                color=best_color,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
            )

    # ensure full k range is visible (start x-axis at 25)
    plt.xlim(25, k)
    plt.ylim(0.6, 1.0)

    plt.xlabel("Top-k")
    plt.ylabel("Score")
    plt.title("HitRate@k")
    # Build legend sorted by score at k=100 (descending)
    sorted_lines = sorted(lines, key=lambda t: t[2][99], reverse=True)
    handles = [t[0] for t in sorted_lines]
    labels = [t[1] for t in sorted_lines]
    custom_text = plt.Line2D([0], [0], color="none", label="Ranking@k=100")
    handles.insert(0, custom_text)
    labels.insert(0, "Ranking@k=100")
    plt.legend(handles=handles, labels=labels, fontsize=16)

    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(50))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))

    # basename = path.split("/")[0]
    outfile = name if name else f"plot_results.png"
    save_path = os.path.join(path, outfile)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--dir", type=str, help="Directory containing experiment results JSON files")
    parser.add_argument("--k", type=int, default=200, help="Value of k for top-k plotting")
    parser.add_argument("--outfile", type=str, help="Output filename for the plot image")
    args = parser.parse_args()

    directory = args.dir
    k = args.k
    outfile = args.outfile

    # Establish what files will be included
    # path = os.path.join(BASE_DIR, directory)
    json_files = get_json_files(directory)
    print(f"Found {len(json_files)} JSON files in {directory}")
    data = make_label_to_data_dict(json_files)

    plot_results(data, path=directory, k=k, name=outfile)


if __name__ == "__main__":
    main()
