import argparse
import h5py
import itertools
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from citeline.database.milvusdb import MilvusDB
from citeline.embedders import Embedder

"""
Takes the given dataset and splits into train/val/test jsonl files. Also
creates embeddings for each split and saves them as h5 files.
"""


def argument_parser():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test and create embeddings.")
    # Add a positional argument for the dataset path
    parser.add_argument("dataset_path", type=str, help="Path to the dataset jsonl file.")

    # Add optional arguments for output base path, train/val/test fractions
    parser.add_argument(
        "--output_base", type=str, default="np_vectors", help="Base path for output files (without extension)."
    )
    parser.add_argument("--train_frac", type=float, default=0.7, help="Fraction of data to use for training set.")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Fraction of data to use for validation set.")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Fraction of data to use for test set.")
    return parser


def split_dataset(df: pd.DataFrame, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    Splits the given DataFrame into train/val/test sets based on the given fractions.
    Returns three DataFrames: train_df, val_df, test_df
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    train_df = df.sample(frac=train_frac, random_state=random_state)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=val_frac / (val_frac + test_frac), random_state=random_state)
    test_df = temp_df.drop(val_df.index)

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, base_path: str):
    """
    Saves the train/val/test DataFrames to jsonl files at the specified base path.
    """
    train_df.to_json(f"{base_path}_train.jsonl", lines=True, orient="records")
    val_df.to_json(f"{base_path}_val.jsonl", lines=True, orient="records")
    test_df.to_json(f"{base_path}_test.jsonl", lines=True, orient="records")


def save_triplets(triplets: np.ndarray, base_path: str):
    """
    Saves the embeddings of the given DataFrame to an h5 file at the specified base path.
    """
    with h5py.File(f"{base_path}_triplets.h5", "w") as h5f:
        h5f.create_dataset("triplets", data=triplets)
    print(f"Saved triplets to {base_path}_triplets.h5")


def get_positive_vectors(example: pd.Series, n: int, db: MilvusDB, collection: str = "qwen06_chunks"):
    """
    precondition: example has 'citation_dois' and 'vector' columns
    """
    assert "citation_dois" in example, "Example must have 'citation_dois' column"
    assert "vector" in example, "Example must have 'vector' column"
    # Get the n closest targets
    target_dois = example["citation_dois"]
    best_positives = []
    for doi in target_dois:
        # Get all records for the target doi
        records = db.select_by_doi(doi, collection_name=collection)
        vectors = np.stack(records["vector"].tolist())

        # Keep the top n most similar by vector
        similarities = np.dot(vectors, example["vector"])
        top_n_similarity_idxs = np.argsort(similarities)[-n:]
        best_positives.append(vectors[top_n_similarity_idxs])

    return np.vstack(best_positives)


def get_negative_vectors(example: pd.Series, n: int, db: MilvusDB, collection: str = "qwen06_chunks"):
    assert "citation_dois" in example, "Example must have 'citation_dois' column"
    assert "vector" in example, "Example must have 'vector' column"

    # Get search results
    search_results = db.search(
        collection_name=collection,
        query_records=[example.to_dict()],
        query_vectors=[example.vector],
        limit=n * 50,  # overfetch to ensure we get enough non-targets
        output_fields=["doi", "vector", "pubdate"],
    )[0]

    target_dois = set(example.citation_dois)
    non_target_vectors = [res["vector"] for res in search_results if res["doi"] not in target_dois]
    if len(non_target_vectors) == 0:
        print(f"Warning: No negative vectors found for example '{example['sent_no_cit'][:30]}...'")
        return np.array([])
    return np.stack(non_target_vectors[:n])


def expand_row_triplets(row: pd.Series) -> np.array:
    query_vector = row["vector"]
    positives = row["positive_vectors"]
    negatives = row["negative_vectors"]
    triplets = [[query_vector, pos, neg] for pos, neg in itertools.product(positives, negatives)]
    return np.array(triplets)


def main():
    parser = argument_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_base = args.output_base
    train_frac = args.train_frac
    val_frac = args.val_frac
    test_frac = args.test_frac

    # Create embedder and db
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create("Qwen/Qwen3-Embedding-0.6B", device=device, normalize=True)
    db = MilvusDB()

    # Load dataset
    df = pd.read_json(dataset_path, lines=True)
    tqdm.pandas(desc="Creating embeddings for dataset")
    df["vector"] = df["sent_no_cit"].progress_apply(lambda x: embedder([x])[0])

    # Split dataset
    train, val, test = split_dataset(df, train_frac, val_frac, test_frac)

    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    for name, df_split in [("train", train), ("val", val), ("test", test)]:
        print(f"=== {name.upper()} SPLIT ===")
        tqdm.pandas(desc="Resolving positive vectors")
        df_split["positive_vectors"] = df_split.progress_apply(
            lambda row: get_positive_vectors(row, n=2, db=db), axis=1
        )

        tqdm.pandas(desc="Resolving negative vectors")
        df_split["negative_vectors"] = df_split.progress_apply(
            lambda row: get_negative_vectors(row, n=2, db=db), axis=1
        )
        tqdm.pandas(desc="Expanding triplets")
        triplets = df_split.progress_apply(expand_row_triplets, axis=1)
        triplets = np.concatenate(triplets.tolist())

        print(f"Total number of triplets: {len(triplets)}")
        save_triplets(triplets, f"{output_base}_{name}")
        df_split.to_json(f"{output_base}_{name}.jsonl", lines=True, orient="records")
        print(f"Saved {name} split to {output_base}_{name}.jsonl")


if __name__ == "__main__":
    main()
