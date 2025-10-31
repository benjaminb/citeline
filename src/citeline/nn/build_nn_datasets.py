from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from itertools import product
import numpy as np
from tqdm import tqdm
import pandas as pd
from citeline.database.milvusdb import MilvusDB
from citeline.embedders import Embedder
from citeline.query_expander import QueryExpander


def embed_sentences(
    df: pd.DataFrame, embedder: Embedder, target_column: str = "sent_no_cit", batch_size: int = 16
) -> pd.DataFrame:
    tqdm.pandas(desc="Embedding sentences")
    df["vector"] = df[target_column].progress_apply(lambda x: embedder([x])[0])
    return df


def get_hard_negatives(
    search_results: list[dict],
    target_dois: set = None,
    n: int = 3,
) -> list[str]:
    """
    Iterates over search results and returns a tensor of the first (hardest) n non-target vectors
    """

    vectors = []
    for rec in search_results:
        if rec["doi"] not in target_dois:
            vectors.append(rec["vector"])
        if len(vectors) == n:
            break
    else:
        raise ValueError("Not enough hard negatives found in search results. Set top k higher?")

    return torch.tensor(vectors)


def get_soft_negatives(
    search_results: list[dict],
    target_dois: set = None,
    n: int = 3,
) -> torch.tensor:
    """
    Assuming search_results > 100, samples n vectors from those results ranked higher than 100
    """

    assert len(search_results) > 100, "search_results must contain more than 100 entries"

    # Filter down to only far results that are not targets
    far_results = search_results[100:]
    far_results = [rec for rec in far_results if rec["doi"] not in target_dois]
    assert len(far_results) > n, "Not enough far results to sample from"

    sampled_results = np.random.choice(far_results, size=n, replace=False)
    vectors = [r["vector"] for r in sampled_results]
    return torch.tensor(vectors)


def get_hard_and_soft_negatives(
    example: pd.Series,
    hard_n: int = 3,
    soft_n: int = 3,
    db: MilvusDB = None,
    collection: str = "qwen06_chunks",
    top_k: int = 500,
) -> tuple[torch.tensor, torch.tensor]:
    results = db.search(
        collection_name=collection,
        query_records=[example.to_dict()],
        query_vectors=[example.vector],
        limit=top_k,
        output_fields=["text", "doi", "pubdate", "citation_count", "vector"],
    )
    results = results[0]  # db.search operates on lists of queries; we only need the first result
    target_dois = set(example.citation_dois)  # All citation DOIs from original query (before explode)
    hard_negatives = get_hard_negatives(results, target_dois=target_dois, n=hard_n)
    soft_negatives = get_soft_negatives(results, target_dois=target_dois, n=soft_n)
    return hard_negatives, soft_negatives


def get_positives(example: pd.Series, n: int = 3, db: MilvusDB = None, collection: str = "qwen06_chunks") -> np.ndarray:
    """
    Returns n positive vectors from the single target DOI, ranked by similarity to the query
    """
    doi = example["target_doi"]  # Single target DOI after explode
    records = db.select_by_doi(doi=doi, collection_name=collection)
    target_vectors = np.array(records["vector"].tolist())

    # Get the top n most similar records
    similarities = example["vector"] @ target_vectors.T
    top_n_indices = np.argpartition(-similarities, min(n, len(similarities)))[:n]
    top_n_vectors = target_vectors[top_n_indices]
    return top_n_vectors


def append_query_expansion(df: pd.DataFrame, expander: str = "add_prev_2", batch_size: int = 32) -> pd.DataFrame:
    """
    Applies the query expansion function to all rows in the df, creating a new df denormalized on sent_no_cit
    """
    expander = QueryExpander(expander, reference_data_path="../../../data/preprocessed/reviews.jsonl")
    df_expanded = df.copy()

    # Apply query expansion in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Batch processing for query expansion"):
        df_expanded.loc[i : i + batch_size, "sent_no_cit"] = expander(df_expanded.loc[i : i + batch_size])
    df_out = pd.concat([df, df_expanded], ignore_index=True)
    return df_out


def data_to_h5(
    df: pd.DataFrame,
    output_path: str,
    embedder: Embedder,
    db: MilvusDB,
    collection: str = "qwen06_chunks",
    num_positives: int = 4,
    num_hard_negatives: int = 4,
    num_soft_negatives: int = 4,
):
    df = embed_sentences(df, embedder)

    queries = []
    positives = []
    hard_negatives = []
    soft_negatives = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        queries.append(row["vector"])
        positives.append(get_positives(row, n=num_positives, db=db, collection=collection))
        hard_negs, soft_negs = get_hard_and_soft_negatives(
            row,
            hard_n=num_hard_negatives,
            soft_n=num_soft_negatives,
            db=db,
            collection=collection,
        )
        hard_negatives.append(hard_negs)
        soft_negatives.append(soft_negs)

    # Convert to numpy arrays
    queries = np.array(queries)
    positives = np.array(positives)
    hard_negatives = np.array(hard_negatives)
    soft_negatives = np.array(soft_negatives)

    # Write to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("queries", data=queries)
        f.create_dataset("positives", data=positives)
        f.create_dataset("hard_negatives", data=hard_negatives)
        f.create_dataset("soft_negatives", data=soft_negatives)

    print(f"Dataset saved to: {output_path}")
    print(f"  Queries: {queries.shape}")
    print(f"  Positives: {positives.shape}")
    print(f"  Hard negatives: {hard_negatives.shape}")
    print(f"  Soft negatives: {soft_negatives.shape}")


def main():
    """
    This script creates a dataset for neural net contrastive learning. The task is to map queries'
    embeddings close to their target documents' embeddings, and simultaneously far from hard and soft negatives.

    """
    EXPANSION = "add_prev_2"
    DATA_FILE = "../../../data/dataset/nontrivial_checked.jsonl"
    N = 1000
    OUTFILE_BASE = "../../../data/dataset/nn"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create("Qwen/Qwen3-Embedding-0.6B", device=device, normalize=True)
    print(f"Using query expansion: {EXPANSION}")
    db = MilvusDB()
    df = pd.read_json(DATA_FILE, lines=True)
    
    # Use this if you want to sample from full dataset
    # df = pd.read_json(DATA_FILE, lines=True).sample(n=N, random_state=42).reset_index(drop=True)

    # Prepare dataframe: append query expansion, explode on citation_dois
    if EXPANSION != "identity":
        df = append_query_expansion(df, expander=EXPANSION)
    df["target_doi"] = df["citation_dois"]
    df = df.explode("target_doi").reset_index(drop=True)

    # Create train/val/test split: 70/15/15
    train_df = df.sample(frac=0.7, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        print(f"=== {split_name.upper()} SET ===")
        print(f"Building {split_name} dataset with {len(split_df)} records...")
        output_path = f"{OUTFILE_BASE}/{split_name}_triplet_ds_{EXPANSION}.h5"
        data_to_h5(
            df=split_df,
            output_path=output_path,
            embedder=embedder,
            db=db,
        )
        df_output_path = f"{OUTFILE_BASE}/{split_name}_ds_{EXPANSION}.jsonl"
        split_df.to_json(df_output_path, orient="records", lines=True)
        print(f"Query dataset saved to: {df_output_path}")


if __name__ == "__main__":
    main()
