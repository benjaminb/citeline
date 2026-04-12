import argparse
import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from citeline.embedders import Embedder
from citeline.nn.config import DatasetConfig
from citeline.query_expander import QueryExpander

tqdm.pandas()

# For query expansion
PATH_TO_REFERENCE = "data/preprocessed/reviews.jsonl"
OUTPUT_DIR = "data/nn_datasets"

"""
Query expansion occurs before embedding so that each string is only embedded once

At the end of this process you have a dataframe with the query string reps, vector reps,
and columns of metadata
"""
# The only columns we need for this task
COLUMNS = [
    "source_doi",
    "sent_no_cit",
    "sent_idx",
    "pubdate",
    "citation_dois",
]


# load dataset
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    missing_cols = set(COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    # Keep only the relevant columns for nn training
    df = df[COLUMNS]
    return df


def preprocess_dataset(df: pd.DataFrame, query_expansions: list[str]) -> pd.DataFrame:
    """Applies all data processing up to the point of embedding the query representatives.
    Currently, this is just computing the query expansions (to get those reps).

    Output df includes columns:
      - source_doi
      - query_expansions: list[str] The query's string representatives
    """
    # Instantiate the query expander(s)
    expanders = [QueryExpander(name, reference_data_path=PATH_TO_REFERENCE) for name in query_expansions]

    # Create a query representatives column, each row has a list of strings for the query reps
    query_expansions = [expander(df) for expander in expanders]  # n x num_expanders
    qe_columns = [list(reps) for reps in zip(*query_expansions)]
    df["query_expansions"] = qe_columns
    return df


def apply_embeddings(df: pd.DataFrame, embedder: str) -> pd.DataFrame:
    # Instantiate the embedder
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create(embedder, device=device, normalize=True)

    # Convert embedder's numpy output to list[list[float]] for parquet compatibility
    df["query_vectors"] = df["query_expansions"].progress_map(lambda x: embedder(x).tolist())
    print("Embeddings applied to all rows.")
    return df


def split_dataset(df: pd.DataFrame, train_frac=0.7, val_frac=0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into train, validation, and test sets."""
    train_df = df.sample(frac=train_frac, random_state=29)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=val_frac / (1 - train_frac), random_state=42)
    test_df = temp_df.drop(val_df.index)
    return train_df, val_df, test_df


# only keep certain columns
# explode on citation_dois
def build_dataset(config_path: str):
    config = DatasetConfig.from_yaml(config_path)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Building dataset with config: {config}")

    # Load and process dataset
    df = load_dataset(config.dataset_path)
    df = preprocess_dataset(df, query_expansions=config.query_expansions)
    df = apply_embeddings(df, embedder=config.embedder)
    train, val, test = split_dataset(df)
    for split_df, name in zip([train, val, test], ["train", "val", "test"]):
        save_path = output_path / f"{name}.parquet"
        split_df.to_parquet(save_path)
        print(f"Saved {name} dataset to {save_path}")


# Get dataset from command line arg
def main():
    parser = argparse.ArgumentParser(description="Build raw datasets for contrastive training.")
    parser.add_argument("--config-path", type=str, help="Path to the dataset config YAML file.")
    args = parser.parse_args()
    build_dataset(args.config_path)


if __name__ == "__main__":
    main()
