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
    filename_prefix: str


@dataclass
class H5DatasetWriterConfig(Config):
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
    dataset_dir: str  # directory containing train.h5, val.h5, test.h5
    dataset_class: str  # The dataset class from contrastive_datasets.py to use; implements picking pos/neg samples
    model: str  # The Adapter subclass from models.py to use as the model architecture
    temperature: float = 0.07
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    epochs: int = 50
    patience: int = 5  # early stopping patience
    checkpoint_path: str = "checkpoints/contrastive_best.pt"


# @dataclass
# class ModelConfig:
#     """Architecture config — two variants supported: 'mlp' and 'residual'."""

#     arch: str  # "mlp" | "residual"
#     hidden_dims: list[int]  # e.g. [2048, 1024] or [1024, 1024, 1024]
#     dropout: float = 0.1

#     @classmethod
#     def from_dict(cls, d: dict) -> "ModelConfig":
#         return cls(**d)


# @dataclass
# class TrainConfig:
#     dataset_dir: str  # directory containing train.h5, val.h5, test.h5
#     model: ModelConfig
#     negative_selection: NegativeSelectionConfig
#     temperature: float = 0.07
#     lr: float = 1e-4
#     weight_decay: float = 1e-2
#     batch_size: int = 256
#     epochs: int = 50
#     patience: int = 5  # early stopping patience
#     checkpoint_path: str = "checkpoints/contrastive_best.pt"

#     @classmethod
#     def from_yaml(cls, path: str) -> "TrainConfig":
#         with open(path) as f:
#             data = yaml.safe_load(f)
#         return cls(
#             dataset_dir=data["dataset_dir"],
#             model=ModelConfig.from_dict(data["model"]),
#             negative_selection=NegativeSelectionConfig.from_dict(data["negative_selection"]),
#             temperature=data.get("temperature", 0.07),
#             lr=data.get("lr", 1e-4),
#             weight_decay=data.get("weight_decay", 1e-2),
#             batch_size=data.get("batch_size", 256),
#             epochs=data.get("epochs", 50),
#             patience=data.get("patience", 5),
#             checkpoint_path=data.get("checkpoint_path", "checkpoints/contrastive_best.pt"),
#         )
