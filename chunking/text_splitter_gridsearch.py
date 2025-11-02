import numpy as np
import os
import pandas as pd
from semantic_text_splitter import TextSplitter
import sys
import torch
from tqdm import tqdm
from citeline.database.milvusdb import MilvusDB
from citeline.embedders import Embedder
import itertools
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
import json
import glob

# Add parent directory to sys.path to import Experiment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiment import Experiment

# Create output directories
os.makedirs("gridsearch", exist_ok=True)
os.makedirs("gridsearch/heatmaps", exist_ok=True)

EMBEDDER_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_DATASET = "../data/dataset/nontrivial_checked.jsonl"
RESEARCH_SAMPLE_DATASET = "gridsearch/temp_research_dataset.jsonl"
N = 1000
TOP_K = 100


def reconstruct_paper(example: pd.Series) -> str:
    return f"{example['title']}\n\nAbstract: {example['abstract']}\n\n{example['body']}"


def load_completed_results(gridsearch_dir="gridsearch"):
    """
    Scan the gridsearch directory for completed results and return:
    1. A set of completed (overlap, min_len, increment) tuples
    2. A dict mapping tuples to their result file paths
    """
    completed = set()
    result_files = {}

    # Find all result JSON files
    pattern = os.path.join(gridsearch_dir, "**", "results_gs_*.json")
    for filepath in glob.glob(pattern, recursive=True):
        # Extract parameters from filename: results_gs_min{min}_inc{inc}_ov{ov}.json
        filename = os.path.basename(filepath)
        try:
            # Parse: results_gs_min50_inc100_ov0.json
            parts = filename.replace("results_gs_", "").replace(".json", "").split("_")
            min_len = int(parts[0].replace("min", ""))
            increment = int(parts[1].replace("inc", ""))
            overlap = int(parts[2].replace("ov", ""))

            param_tuple = (overlap, min_len, increment)
            completed.add(param_tuple)
            result_files[param_tuple] = filepath
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
            continue

    return completed, result_files


def chunk_text(text: str, splitter: TextSplitter) -> list[str]:
    chunks = splitter.chunks(text)
    chunks = [stripped_chunk.replace("\x00", "") for chunk in chunks if (stripped_chunk := chunk.strip())]
    return chunks


def create_heatmap(heatmap_data, min_lengths, increments, overlap_val, k_value, output_path):
    """Create and save a heatmap for the given data."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap with discrete cells
    im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto", origin="lower")

    # Set tick labels to actual parameter values
    ax.set_xticks(np.arange(len(min_lengths)))
    ax.set_xticklabels(min_lengths)
    ax.set_yticks(np.arange(len(increments)))
    ax.set_yticklabels(increments)

    # Labels
    ax.set_xlabel("Min Length", fontsize=12)
    ax.set_ylabel("Increment (Max = Min + Inc)", fontsize=12)
    ax.set_title(f"Hitrate@{k_value} (Overlap={overlap_val})", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Hitrate@{k_value}", fontsize=12)

    # Add grid lines to emphasize discrete cells
    ax.set_xticks(np.arange(len(min_lengths)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(increments)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

    # Add text annotations showing exact values
    for i in range(len(increments)):
        for j in range(len(min_lengths)):
            ax.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    sample_df = pd.read_json(QUERY_DATASET, lines=True).sample(n=N, random_state=42)
    target_dois = set(sample_df["citation_dois"].explode().unique())
    print(f"Testing text splitter grid search on {len(target_dois)} unique DOIs")

    # Set up embedder and db
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create(EMBEDDER_NAME, device=device, normalize=True)
    tqdm.pandas(desc="Embedding sample queries")
    sample_df["vector"] = sample_df.progress_apply(lambda row: embedder([row["sent_no_cit"]])[0], axis=1)
    db = MilvusDB()

    # Save the sampled queries to a temporary file so experiments use the correct subset
    sample_queries_path = "gridsearch/temp_sample_queries.jsonl"
    sample_df.to_json(sample_queries_path, orient="records", lines=True)
    print(f"Saved {len(sample_df)} sampled queries to {sample_queries_path}")

    # Prep the full dataset (research papers)
    if not os.path.exists(RESEARCH_SAMPLE_DATASET):
        print(f"Preparing research dataset sample at {RESEARCH_SAMPLE_DATASET}...")
        research = pd.read_json("../data/preprocessed/research.jsonl", lines=True)

        # Research in targets: needed for recall
        research_targets = research[research["doi"].isin(target_dois)]  # Filter only to DOIs in the target set

        # Add papers that are not targets (helps measure precision)
        research_distractors = research[~research["doi"].isin(target_dois)]
        research_distractors = research_distractors.sample(frac=0.2, random_state=42)

        research = pd.concat([research_targets, research_distractors], ignore_index=True).drop_duplicates(
            subset=["doi"]
        )
        research["paper"] = research.apply(reconstruct_paper, axis=1)
        research["pubdate"] = research["pubdate"].apply(lambda x: int(x.replace("-", "")))

        # Filter to only needed columns
        cols_to_keep = ["doi", "citation_count", "pubdate", "paper"]
        research = research[cols_to_keep]

        research.to_json("gridsearch/temp_research_dataset.jsonl", orient="records", lines=True)
        print(f"Saved research dataset sample with {len(research)} papers")
    else:
        print(f"Loading existing research dataset sample from {RESEARCH_SAMPLE_DATASET}...")
        research = pd.read_json(RESEARCH_SAMPLE_DATASET, lines=True)
        print(f"Loaded research dataset sample with {len(research)} papers")

    overlap_stepsize = 50
    min_lengths = np.arange(100, 1001, 200)
    increments = np.arange(100, 1001, 200)
    overlaps = np.arange(0, 201, overlap_stepsize)

    hitrate_results = np.zeros((len(overlaps), len(min_lengths), len(increments), TOP_K))
    recall_results = np.zeros((len(overlaps), len(min_lengths), len(increments), TOP_K))
    iou_results = np.zeros((len(overlaps), len(min_lengths), len(increments), TOP_K))

    # Map param values to indices for results
    overlap_to_idx = {v: i for i, v in enumerate(overlaps)}
    min_len_to_idx = {v: i for i, v in enumerate(min_lengths)}
    inc_to_idx = {v: i for i, v in enumerate(increments)}

    # Check for completed results to resume from
    completed_params, result_files = load_completed_results("gridsearch")
    print(f"Found {len(completed_params)} completed experiments, will skip these")

    # Load existing results into arrays
    for param_tuple, filepath in result_files.items():
        overlap, min_len, increment = param_tuple
        try:
            with open(filepath, "r") as f:
                result_data = json.load(f)

            ov_idx = overlap_to_idx[overlap]
            min_len_idx = min_len_to_idx[min_len]
            inc_idx = inc_to_idx[increment]

            hitrate_results[ov_idx, min_len_idx, inc_idx] = np.array(result_data["average_hitrate_at_k"])
            recall_results[ov_idx, min_len_idx, inc_idx] = np.array(result_data["average_recall_at_k"])
            iou_results[ov_idx, min_len_idx, inc_idx] = np.array(result_data["average_iou_at_k"])
        except Exception as e:
            print(f"Warning: Could not load results from {filepath}: {e}")

    # Grid search over parameters with progress tracking
    total_combinations = len(overlaps) * len(min_lengths) * len(increments)
    for overlap, min_len, increment in tqdm(
        itertools.product(overlaps, min_lengths, increments), total=total_combinations, desc="Grid search progress"
    ):
        # Skip invalid parameter combinations where overlap is too large
        if overlap > min_len - overlap_stepsize:
            continue

        # Skip if already completed
        if (overlap, min_len, increment) in completed_params:
            continue

        # Create temp df to insert into DB
        temp_df = research.copy()

        # Set up splitter with current params
        splitter = TextSplitter((min_len, min_len + increment), overlap=overlap, trim=True)

        # Chunk the papers, prep the df for insertion
        with tqdm(
            total=len(temp_df),
            desc=f"Chunking papers (min_len={min_len}, max_len={min_len + increment}, overlap={overlap})",
        ) as pbar:
            temp_df["text"] = temp_df["paper"].apply(lambda x: (pbar.update(1), chunk_text(x, splitter))[1])

        # Ensure tqdm output is fully flushed before any subsequent prints
        sys.stdout.flush()

        temp_df.drop(columns=["paper"], inplace=True)
        temp_df = temp_df.explode("text")

        # Have MilvusDB embed and insert
        collection_name = f"temp_min{min_len}_inc{increment}_ov{overlap}"
        db.create_vector_collection_pd(
            name=collection_name, data=temp_df, embedder_name=EMBEDDER_NAME, normalize=True, batch_size=16
        )

        # Create experiment config (convert numpy types to Python types for JSON serialization)
        config = {
            "dataset": sample_queries_path,  # Use the sampled queries, not the full dataset!
            "table": collection_name,
            "target_column": "vector",
            "metric": "IP",
            "embedder": EMBEDDER_NAME,
            "normalize": True,
            "query_expansion": "identity",
            "batch_size": 16,
            "top_k": TOP_K,
            "output_path": f"gridsearch/",
            "plot_label": f"Len: ({min_len}, {min_len + increment}), Overlap: {overlap}",
            "experiment_name": f"gs_min{min_len}_inc{increment}_ov{overlap}",
            "min_length": int(min_len),
            "increment": int(increment),
            "overlap": int(overlap),
            "has_precomputed_embeddings": True,  # Embeddings already computed above
        }

        # Run an experiment using the sample_df as query data and this collection as target table
        experiment = Experiment(**config)
        experiment.run()

        # Store results
        ov_idx = overlap_to_idx[overlap]
        min_len_idx = min_len_to_idx[min_len]
        inc_idx = inc_to_idx[increment]
        hitrate_results[ov_idx, min_len_idx, inc_idx] = np.array(experiment.avg_hitrate_at_k)
        recall_results[ov_idx, min_len_idx, inc_idx] = np.array(experiment.avg_recall_at_k)
        iou_results[ov_idx, min_len_idx, inc_idx] = np.array(experiment.avg_iou_at_k)

        # Cleanup
        db.drop_collection(collection_name)

    # Save results to npy files
    np.save("gridsearch/text_splitter_gridsearch_hitrate.npy", hitrate_results)
    np.save("gridsearch/text_splitter_gridsearch_recall.npy", recall_results)
    np.save("gridsearch/text_splitter_gridsearch_iou.npy", iou_results)

    # Save parameter info for later analysis
    np.savez(
        "gridsearch/parameters.npz", overlaps=overlaps, min_lengths=min_lengths, increments=increments, top_k=TOP_K
    )

    # After saving results, before creating heatmaps
    k_idx = 49  # hitrate@50
    best_idx = np.unravel_index(np.argmax(hitrate_results[:, :, :, k_idx]), hitrate_results[:, :, :, k_idx].shape)
    best_overlap = overlaps[best_idx[0]]
    best_min_len = min_lengths[best_idx[1]]
    best_increment = increments[best_idx[2]]
    best_hitrate = hitrate_results[best_idx[0], best_idx[1], best_idx[2], k_idx]

    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS FOR HITRATE@50:")
    print(f"  Overlap: {best_overlap}")
    print(f"  Min Length: {best_min_len}")
    print(f"  Increment: {best_increment}")
    print(f"  Max Length: {best_min_len + best_increment}")
    print(f"  Hitrate@50: {best_hitrate:.4f}")
    print(f"{'='*60}\n")

    for k_value in [50, 100]:
        k_idx = k_value - 1  # k=50 is at index 49
        for ov_idx, overlap_val in enumerate(overlaps):
            heatmap_data = hitrate_results[ov_idx, :, :, k_idx]
            output_path = f"gridsearch/heatmaps/hitrate{k_value}_overlap{overlap_val}.png"
            create_heatmap(heatmap_data, min_lengths, increments, overlap_val, k_value, output_path)


if __name__ == "__main__":
    main()
