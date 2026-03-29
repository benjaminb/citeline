import pandas as pd
import torch
from tqdm import tqdm
from citeline.embedders import Embedder
from citeline.nn.config import DatasetConfig
from citeline.query_expander import QueryExpander

tqdm.pandas()

# For query expansion
PATH_TO_REFERENCE = "data/preprocessed/reviews.jsonl"

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
    df = df.dropna(subset=COLUMNS)
    return df


def preprocess_dataset(df: pd.DataFrame, query_expansions: list[str]) -> pd.DataFrame:
    """Applies query expansions and explodes on citation_dois to get one row per (query, target_doi).
    Output df includes columns: 
      - source_doi
      - query_expansions: list[str] The query's string representatives
      - citation_doi (one per row after explode)
    """
    # Instantiate the query expander(s)
    expanders = [QueryExpander(name, reference_data_path=PATH_TO_REFERENCE) for name in query_expansions]

    # Create a query representatives column, each row has a list of strings for the query reps
    query_expansions = [expander(df) for expander in expanders] # n x num_expanders
    qe_columns = [list(reps) for reps in zip(*query_expansions)]
    df["query_expansions"] = qe_columns

    # Explode on targets to get one row per (query, target_doi)
    df = df.explode("citation_dois").reset_index(drop=True)
    df = df.rename({"citation_dois": "citation_doi"}, axis=1)
    return df

def apply_embeddings(df: pd.DataFrame, embedder: str) -> pd.DataFrame:

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"    
    embedder = Embedder.create(embedder, device=device, normalize=True)
    print(f"Applying embedder {embedder.model_name} to {len(df)} rows")
    df["embeddings"] = df["query_expansions"].progress_map(embedder)
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
    
    # Load and process dataset
    df = load_dataset(config.dataset_path)
    df = preprocess_dataset(df, query_expansions=config.query_expansions)
    df = apply_embeddings(df, embedder=config.embedder)

    print(f"Length: {len(df)}")
    print(df.iloc[1]["query_expansions"])
