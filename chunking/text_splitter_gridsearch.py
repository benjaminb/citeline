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

EMBEDDER_NAME = "Qwen/Qwen3-Embedding-0.6B"
# TODO: run this with bigger dataset later
QUERY_DATASET = "../data/dataset/nontrivial_10.jsonl"
TOP_K = 1000


def reconstruct_paper(example: pd.Series) -> str:
    return f"{example['title']}\n\nAbstract: {example['abstract']}\n\n{example['body']}"


def chunk_text(text: str, splitter: TextSplitter) -> list[str]:
    chunks = splitter.chunks(text)
    chunks = [stripped_chunk.replace("\x00", "") for chunk in chunks if (stripped_chunk := chunk.strip())]
    return chunks


def main():
    sample_df = pd.read_json("../data/dataset/nontrivial_checked.jsonl", lines=True).sample(n=10, random_state=42)
    target_dois = set(sample_df["citation_dois"].explode().unique())
    print(f"Testing text splitter grid search on {len(target_dois)} unique DOIs")

    # Set up embedder and db
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create(EMBEDDER_NAME, device=device, normalize=True)
    tqdm.pandas(desc="Embedding sample queries")
    sample_df["vector"] = sample_df.progress_apply(lambda row: embedder([row["sent_no_cit"]])[0], axis=1)
    db = MilvusDB()

    # Prep the full dataset (research papers)
    research = pd.read_json("../data/preprocessed/research.jsonl", lines=True)
    research = research[research["doi"].isin(target_dois)]
    research["paper"] = research.apply(reconstruct_paper, axis=1)
    research["pubdate"] = research["pubdate"].apply(lambda x: int(x.replace("-", "")))

    # Filter to only needed columns
    cols_to_keep = ["doi", "citation_count", "pubdate", "paper"]
    research = research[cols_to_keep]

    # min_lengths = np.arange(50, 1001, 50)
    # increments = np.arange(50, 1001, 50)
    # overlaps = np.arange(0, 201, 25)

    """
    Shorter spans for testing
    """
    overlaps = np.arange(0, 51, 25)
    min_lengths = np.arange(50, 101, 50)
    increments = np.arange(50, 101, 50)

    hitrate_results = np.zeros((len(overlaps), len(min_lengths), len(increments)), TOP_K)
    recall_results = np.zeros((len(overlaps), len(min_lengths), len(increments)), TOP_K)

    # Map param values to indices for results
    overlap_to_idx = {v: i for i, v in enumerate(overlaps)}
    min_len_to_idx = {v: i for i, v in enumerate(min_lengths)}
    inc_to_idx = {v: i for i, v in enumerate(increments)}

    # Grid search over parameters
    for overlap, min_len, increment in itertools.product(
        overlaps,
        min_lengths,
        increments,
    ):
        # Create temp df to insert into DB
        temp_df = research.copy()

        # Set up splitter with current params
        overlap = min(min_len - 1, overlap)  # ensure overlap is not greater than min_len
        splitter = TextSplitter((min_len, min_len + increment), overlap=overlap, trim=True)

        # Chunk the papers, prep the df for insertion
        tqdm.pandas(desc=f"Chunking papers (min_len={min_len}, max_len={min_len + increment}, overlap={overlap})")
        temp_df["text"] = temp_df["paper"].progress_apply(lambda x: chunk_text(x, splitter))
        temp_df.drop(columns=["paper"], inplace=True)
        temp_df = temp_df.explode("text")

        # Have MilvusDB embed and insert
        collection_name = f"temp_min{min_len}_inc{increment}_ov{overlap}"
        db.create_vector_collection_pd(
            name=collection_name, data=temp_df, embedder_name=EMBEDDER_NAME, normalize=True, batch_size=16
        )

        # Create experiment config
        config = {
            "dataset": QUERY_DATASET,
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
            "min_length": min_len,
            "increment": increment,
            "overlap": overlap,
        }

        # Run an experiment using the sample_df as query data and this collection as target table
        experiment = Experiment(**config)
        experiment.run()

        # Use qwen 06
        ov_idx = overlap_to_idx[overlap]
        min_len_idx = min_len_to_idx[min_len]
        inc_idx = inc_to_idx[increment]
        avg_hitrate_at_k = np.array(experiment.results["average_hitrate_at_k"])
        hitrate_results[ov_idx, min_len_idx, inc_idx] = avg_hitrate_at_k
        recall_results[ov_idx, min_len_idx, inc_idx] = np.array(experiment.results["average_recall_at_k"])

        # Cleanup
        db.drop_collection(collection_name)
    # Save results to npy file
    np.save("text_splitter_gridsearch_results.npy", hitrate_results)
    np.save("text_splitter_gridsearch_recall_results.npy", recall_results)


if __name__ == "__main__":
    main()
