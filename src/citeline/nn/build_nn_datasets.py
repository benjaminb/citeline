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


def get_hard_records(
    example: pd.Series, n: int = 4, db: MilvusDB = None, collection: str = "qwen06_chunks"
) -> list[str]:
    """
    Overfetches 10*n most similar records (bc if two reps from same doc are in top n, we won't have n distinct non-target dois)

    Returns:
      A list of doi's, ordered by their max similarity to the query
    """
    results = db.search(
        collection_name=collection,
        query_records=[example.to_dict()],
        query_vectors=[example.vector],
        limit=10 * n,
        output_fields=["text", "doi", "pubdate", "citation_count", "vector"],
    )
    results = results[0]  # db.search operates on lists of queries; we only need the first result

    # Filter results to non-targets only, return the first n
    target_dois = set(example.citation_dois)
    non_target_results = [r for r in results if r["doi"] not in target_dois][:n]
    vectors = [r["vector"] for r in non_target_results]
    # return np.array(vectors)
    return torch.tensor(vectors)


def get_similar_targets(
    example: pd.Series, n: int = 4, db: MilvusDB = None, collection: str = "qwen06_chunks"
) -> list[str]:
    """
    Returns:
      A list of doi's, ordered by their max similarity to the query
    """

    def get_similar_target_vectors_by_doi(doi: str) -> np.ndarray:
        records = db.select_by_doi(doi=doi, collection_name=collection)
        target_vectors = np.array(records["vector"].tolist())

        # Get the top n most similar records
        similarities = example["vector"] @ target_vectors.T
        top_n_indices = np.argpartition(-similarities, n)[:n]
        top_n_vectors = target_vectors[top_n_indices]
        return top_n_vectors

    blocks = []
    for doi in example["citation_dois"]:
        blocks.append(get_similar_target_vectors_by_doi(doi))
    return np.vstack(blocks)


def append_query_expansion(df: pd.DataFrame, expander: str = "add_prev_2", batch_size: int = 32) -> pd.DataFrame:
    """
    Applies the query expansion function to all rows in the df, creating a new df denormalized on sent_no_cit
    """
    expander = QueryExpander(expander, reference_data_path="../../../data/preprocessed/reviews.jsonl")
    df_expanded = df.copy()

    # Apply query expansion in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Batch processing for query expansion"):
        # end_idx = min(i + batch_size, len(df))
        df_expanded.loc[i : i + batch_size, "sent_no_cit"] = expander(df_expanded.loc[i : i + batch_size])
    df_out = pd.concat([df, df_expanded], ignore_index=True)
    return df_out


def data_to_h5(df: pd.DataFrame, output_path: str, embedder: Embedder, db: MilvusDB, collection: str = "qwen06_chunks"):
    df = embed_sentences(df, embedder)
    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset("triplets", shape=(0, 3, 1024), maxshape=(None, 3, 1024), dtype="float32")
        triplet_buffer = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating triplets"):
            query_vector = row["vector"]
            positive_vectors = get_similar_targets(row, n=4, db=db, collection=collection)
            negative_vectors = get_hard_records(row, n=4, db=db, collection=collection)

            for pos_vec, neg_vec in product(positive_vectors, negative_vectors):
                triplet_buffer.append(np.array([query_vector, pos_vec, neg_vec]))

            if len(triplet_buffer) >= 1024:
                new_size = dset.shape[0] + len(triplet_buffer)
                dset.resize(new_size, axis=0)
                dset[-len(triplet_buffer) :] = np.array(triplet_buffer)
                triplet_buffer = []

        # Write any remaining triplets in the buffer
        if triplet_buffer:
            new_size = dset.shape[0] + len(triplet_buffer)
            dset.resize(new_size, axis=0)
            dset[-len(triplet_buffer) :] = np.array(triplet_buffer)

        num_samples = dset.shape[0]
        print(f"Dataset (n={num_samples}) save to: {output_path}")  # (num_samples, 3, vector_dim)


def main():
    """
    This script creates a dataset for neural net contrastive learning. The task is to map queries'
    embeddings close to their target documents' embeddings, and simultaneously far from hard negatives.

    This creates a triplet dataset of numpy arrays [N * 3 * D], where each triplet is:
      (query_vector, positive_vector, hard_vector)

    From each original sentence in the dataset, we create multiple triplets by:
        1. Representing the original sentence as-is plus with add_prev_2 query expansion, since this is a top-performing strategy
        2. Retrieve the top 4 chunks by similarity and the top 4 hard negatives by similarity
        3. Create all combinations of (query, positive, hard) triplets from these
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create("Qwen/Qwen3-Embedding-0.6B", device=device, normalize=True)
    db = MilvusDB()
    df = pd.read_json("../../../data/dataset/nontrivial_checked.jsonl", lines=True)
    df = append_query_expansion(df, expander="add_prev_2")

    # Create train/val/test split: 70/15/15
    train_df = df.sample(frac=0.7, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        print(f"=== {split_name.upper()} SET ===")
        print(f"Building {split_name} dataset with {len(split_df)} records...")
        output_path = f"../../../data/dataset/{split_name}_nn_triplet_dataset.h5"
        data_to_h5(
            df=split_df,
            output_path=output_path,
            embedder=embedder,
            db=db,
        )


if __name__ == "__main__":
    main()
