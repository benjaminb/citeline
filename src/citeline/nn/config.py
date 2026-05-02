from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import yaml


class Config(ABC):

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class DatasetConfig(Config):
    dataset_path: str  # Path to jsonl file
    embedder: str
    db_collection: str  # Name of the Milvus collection relevant to the chosen embedder
    query_expansions: list[str]
    output_path: str


@dataclass
class ContrastiveDatasetBuilderConfig(Config):
    dataset_dir: str  # Directory containing train.parquet, val.parquet, test.parquet
    output_dir: str  # Directory to write train.h5, val.h5, test.h5
    strategy: str
    adapter: str

    # Number of positive vectors PER TARGET to include
    num_positives: int

    # Number of negative vectors PER ANCHOR to include
    num_negatives: int


@dataclass
class TrainConfig(Config):
    parquet_datadir: str
    h5_datadir: str  # directory containing train.h5, val.h5, test.h5
    strategy: str
    num_positives: int
    num_negatives: int
    dataset_class: str  # The dataset class from contrastive_datasets.py to use; implements picking pos/neg samples
    model: str  # The Adapter subclass from models.py to use as the model architecture
    loss: str  # The ContrastiveLossFunction subclass
    loss_schedule: Optional[str] = None  # The LossSchedule subclass to use for weighting pos/neg losses during training
    temperature: float = 0.07
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    epochs: int = 50
    rebuild_patience: int = 3  # num epochs with no train margin improvement -> rebuild dataset
    checkpoint_path: str = "checkpoints/contrastive_best.pt"
