import pandas as pd
from citeline.nn.config import DatasetConfig

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
    df = df.dropna(subset=COLUMNS)
    df = df.explode("citation_dois").reset_index(drop=True)
    df = df.rename({'citation_dois': 'citation_doi'}, axis=1)
    return df

# only keep certain columns
# explode on citation_dois
def build_dataset():
    config = DatasetConfig.from_yaml("src/citeline/nn/configs/test_dataset_config.yaml")
    df = load_dataset(config.dataset_path)
    print(f"Length: {len(df)}")
    print(df.iloc[0])