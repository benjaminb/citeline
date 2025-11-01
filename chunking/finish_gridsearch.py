"""
Fill in missing grid search results for:
- overlap = 0
- min_length = 750
- increment = 650, 700, 750, 800, 850, 900, 950, 1000
"""

# Set matplotlib to use non-interactive backend to avoid tkinter issues
import matplotlib

matplotlib.use("Agg")

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

# Add parent directory to sys.path to import Experiment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiment import Experiment

# Create output directories
os.makedirs("gridsearch", exist_ok=True)

EMBEDDER_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_DATASET = "../data/dataset/nontrivial_checked.jsonl"
N = 100
TOP_K = 200


def reconstruct_paper(example: pd.Series) -> str:
    return f"{example['title']}\n\nAbstract: {example['abstract']}\n\n{example['body']}"


def chunk_text(text: str, splitter: TextSplitter) -> list[str]:
    chunks = splitter.chunks(text)
    chunks = [stripped_chunk.replace("\x00", "") for chunk in chunks if (stripped_chunk := chunk.strip())]
    return chunks


def check_completed(overlap, min_len, increment, gridsearch_dir="gridsearch"):
    """Check if this experiment has already been completed."""
    dir_name = f"gs_min{min_len}_inc{increment}_ov{overlap}"
    result_file = os.path.join(gridsearch_dir, dir_name, f"results_{dir_name}.json")
    return os.path.exists(result_file)


def main():
    # Parameters to fill in
    overlap = 0
    min_len = 750
    increments = np.arange(850, 1001, 50)  # 850, 900, 950, 1000

    print(f"Fill-in Grid Search Parameters:")
    print(f"  Overlap: {overlap}")
    print(f"  Min Length: {min_len}")
    print(f"  Increments: {list(increments)}")
    print()

    # Check which ones are already completed
    to_run = []
    already_done = []

    for increment in increments:
        if check_completed(overlap, min_len, increment):
            already_done.append(increment)
            print(f"✓ Already completed: min={min_len}, inc={increment}, ov={overlap}")
        else:
            to_run.append(increment)
            print(f"✗ Missing: min={min_len}, inc={increment}, ov={overlap}")

    print()
    print(f"Summary: {len(already_done)} already completed, {len(to_run)} to run")

    if not to_run:
        print("All experiments already completed!")
        return

    print(f"\nWill run {len(to_run)} experiments: {to_run}")
    print()

    # Load and prepare data
    sample_df = pd.read_json(QUERY_DATASET, lines=True).sample(n=N, random_state=42)
    target_dois = set(sample_df["citation_dois"].explode().unique())
    print(f"Testing on {len(target_dois)} unique DOIs from {N} sample queries")

    # Set up embedder and db
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create(EMBEDDER_NAME, device=device, normalize=True)
    tqdm.pandas(desc="Embedding sample queries")
    sample_df["vector"] = sample_df.progress_apply(lambda row: embedder([row["sent_no_cit"]])[0], axis=1)
    db = MilvusDB()

    # Save the sampled queries to a temporary file
    sample_queries_path = "gridsearch/temp_sample_queries.jsonl"
    sample_df.to_json(sample_queries_path, orient="records", lines=True)
    print(f"Saved {len(sample_df)} sampled queries to {sample_queries_path}")

    # Prep the full dataset (research papers)
    research = pd.read_json("../data/preprocessed/research.jsonl", lines=True)
    research = research[research["doi"].isin(target_dois)]
    research["paper"] = research.apply(reconstruct_paper, axis=1)
    research["pubdate"] = research["pubdate"].apply(lambda x: int(x.replace("-", "")))

    # Filter to only needed columns
    cols_to_keep = ["doi", "citation_count", "pubdate", "paper"]
    research = research[cols_to_keep]

    # Run experiments for missing combinations
    for increment in tqdm(to_run, desc="Running missing experiments"):
        print(f"\n{'='*80}")
        print(f"Running: min={min_len}, inc={increment}, overlap={overlap}")
        print(f"  Max length: {min_len + increment}")
        print(f"{'='*80}\n")

        # Create temp df to insert into DB
        temp_df = research.copy()

        # Set up splitter with current params
        splitter = TextSplitter((min_len, min_len + increment), overlap=overlap, trim=True)

        # Chunk the papers
        with tqdm(total=len(temp_df), desc=f"Chunking papers") as pbar:
            temp_df["text"] = temp_df["paper"].apply(lambda x: (pbar.update(1), chunk_text(x, splitter))[1])

        # Ensure tqdm output is fully flushed
        sys.stdout.flush()

        temp_df.drop(columns=["paper"], inplace=True)
        temp_df = temp_df.explode("text")

        # Have MilvusDB embed and insert
        collection_name = f"temp_min{min_len}_inc{increment}_ov{overlap}"
        db.create_vector_collection_pd(
            name=collection_name, data=temp_df, embedder_name=EMBEDDER_NAME, normalize=True, batch_size=16
        )

        # Create experiment config
        config = {
            "dataset": sample_queries_path,
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
            "has_precomputed_embeddings": True,
        }

        # Run experiment
        experiment = Experiment(**config)
        experiment.run()

        # Cleanup
        db.drop_collection(collection_name)

        print(f"\n✓ Completed: min={min_len}, inc={increment}, ov={overlap}")
        print(f"  Hitrate@50: {experiment.avg_hitrate_at_k[49]:.4f}")
        print(f"  Hitrate@100: {experiment.avg_hitrate_at_k[99]:.4f}")

    print(f"\n{'='*80}")
    print(f"All {len(to_run)} missing experiments completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
