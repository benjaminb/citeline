import matplotlib.pyplot as plt
import json

PATH_TO_DATA = "../data/summaries/summary_results.json"

"""
{
    "0.6B": {
        "chunks (baseline)": {
            "hitrate@25": 0.6956,
            "recall@25": 0.6486,
            "hitrate@50": 0.7601,
            "recall@50": 0.7177
        },
        "chunks+ADD-02": {
            "hitrate@25": 0.7489,
            "recall@25": 0.6996,
            "hitrate@50": 0.8150,
            "recall@50": 0.7728
        },
        "chunks+ADD-02+RRF": {
            "hitrate@25": 0.7647,
            "recall@25": 0.7161,
            "hitrate@50": 0.8284,
            "recall@50": 0.7869
        }
    },
    "8B": {
        "chunks (baseline)": {
            "hitrate@25": 0.7570,
            "recall@25": 0.7119,
            "hitrate@50": 0.8126,
            "recall@50": 0.7757
        },
        "chunks+ADD-02": {
            "hitrate@25": 0.7974,
            "recall@25": 0.7494,
            "hitrate@50": 0.8584,
            "recall@50": 0.8204
        },
        "chunks+ADD-02+RRF": {
            "hitrate@25": 0.8065,
            "recall@25": 0.7580,
            "hitrate@50": 0.8607,
            "recall@50": 0.8229
        }
    }
}
"""

with open(PATH_TO_DATA, "r") as f:
    results = json.load(f)

# Metrics to plot
metrics = ["hitrate@25", "recall@25", "hitrate@50", "recall@50"]
metric_labels = {
    "hitrate@25": "Hitrate@25",
    "recall@25": "Recall@25",
    "hitrate@50": "Hitrate@50",
    "recall@50": "Recall@50",
}
model_order = ["chunks (baseline)", "chunks+ADD-02", "chunks+ADD-02+RRF"]

for model_size in ["0.6B", "8B"]:
    plt.figure(figsize=(8, 5))
    color_map = {
        "hitrate@25": "#1f77b4",  # regular blue (matplotlib default)
        "hitrate@50": "#aec7e8",  # light blue
        "recall@25": "#ff7f0e",  # regular orange (matplotlib default)
        "recall@50": "#ffbb78",  # light orange
    }
    marker_map = {
        "hitrate@25": "s",  # square
        "hitrate@50": "s",  # square
        "recall@25": "D",  # diamond
        "recall@50": "D",  # diamond
    }
    x_axis_labels = {
        "chunks (baseline)": "Chunks\n(Baseline)",
        "chunks+ADD-02": "Chunks\n+ ADD-02",
        "chunks+ADD-02+RRF": "Chunks\n+ ADD-02 + RRF",
    }
    for metric in metrics:
        y = [results[model_size][model][metric] for model in model_order]
        plt.plot(
            model_order,
            y,
            marker=marker_map.get(metric, "o"),
            label=metric_labels[metric],
            color=color_map.get(metric, None),
        )
    # Apply custom x-axis labels with styling
    custom_labels = [x_axis_labels[model] for model in model_order]
    plt.xticks(
        ticks=range(len(model_order)),
        labels=custom_labels,
        fontsize=14,
        # fontweight="bold",
        # rotation=10,
        # fontname="DejaVu Sans",
    )
    plt.title(f"Summary Metrics for Qwen3 {model_size}", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.6, 0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"summary_{model_size.replace('.', '_')}.png", bbox_inches="tight", pad_inches=0.1)
