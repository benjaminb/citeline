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
"""

BASE_DIR = "experiments/"


def get_json_files(path):
    return glob.glob(os.path.join(path, "**/*.json"), recursive=True)


def path_to_label(path):
    # Convert a file path to a label by removing the base directory and file extension
    filename = os.path.basename(path)
    pieces = filename.split("_")
    model = pieces[2]
    document_rep = pieces[3]
    query_expansion = ""
    if "title" in filename:
        query_expansion = "title"
    elif "abstract" in filename:
        query_expansion = "abstract"
    elif "prev_3" in filename:
        query_expansion = "prev_3"
    elif "prev_5" in filename:
        query_expansion = "prev_5"
    elif "prev_7" in filename:
        query_expansion = "prev_7"

    # Filter out empty strings before joining
    parts = [model, document_rep, query_expansion]
    return "+".join([part for part in parts if part and not part in ["basic"]])


def make_label_to_data_dict(json_files: list[str]) -> dict:
    data = dict()
    for file in json_files:
        results = json.load(open(file))
        hitrates = results["average_hitrate_at_k"]
        label = path_to_label(file)
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
            k_values, hitrates_trunc, drawstyle="steps-post", linestyle="-", lw=1.5, alpha=0.8, label=label
        )
        lines.append((line, label, hitrates_trunc))
        # label slightly to the right of k=200 (or at k if k<200)
        plt.text(
            label_x,
            y_label,
            label,
            color=line.get_color(),
            va="center",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    # ensure full k range is visible
    plt.xlim(1, k)
    plt.xlabel("Top-k")
    plt.ylabel("Score")
    plt.title("HitRate@k")
    # Build legend sorted by score at k=200 (descending)
    sorted_lines = sorted(lines, key=lambda t: t[2][-1], reverse=True)
    handles = [t[0] for t in sorted_lines]
    labels = [t[1] for t in sorted_lines]
    # plt.legend(handles=handles, labels=labels)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(100))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))

    basename = path.split("/")[0]
    outfile = name if name else f"{basename}_results.png"
    save_path = os.path.join(BASE_DIR, path, outfile)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--dir", type=str, help="Directory containing experiment results JSON files")
    parser.add_argument("--k", type=int, help="Value of k for top-k plotting")
    parser.add_argument("--outfile", type=str, help="Output filename for the plot image")
    args = parser.parse_args()

    directory = args.dir if args.dir else sys.argv[1]
    k = args.k if args.k else int(sys.argv[2]) if sys.argv[2] else 1000
    outfile = args.outfile if args.outfile else sys.argv[3] if sys.argv[3] else None

    # Establish what files will be included
    path = os.path.join(BASE_DIR, directory)
    json_files = get_json_files(path)
    print(f"Found {len(json_files)} JSON files in {path}")
    data = make_label_to_data_dict(json_files)

    plot_results(data, path=directory, k=k, name=outfile)


if __name__ == "__main__":
    main()
