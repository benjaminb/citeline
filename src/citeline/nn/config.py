from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class DatasetConfig:
    # Path to jsonl file
    dataset_path: str

    embedder: str

    # Name of the Milvus collection relevant to the chosen embedder
    db_collection: str

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

@dataclass
class BuildConfig:
    dataset_source: str
    embedder: str
    collection: str
    num_negatives: int
    milvus_top_k: int
    output_dir: str
    normalize: bool = True
    train_frac: float = 0.70
    val_frac: float = 0.15

    @classmethod
    def from_yaml(cls, path: str) -> "BuildConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class NegativeSelectionConfig:
    """Controls how negatives are selected and weighted from the HDF5 dataset at train time."""
    num_negatives: int              # how many negatives per sample during training
    rank_range: list[int]           # [lo, hi) slice into stored K negatives; e.g. [0, 50] = hardest
    weight_scheme: str              # "uniform" | "bins" | "cosine_sim" | "inv_rank"

    # bins params
    num_bins: int = 4
    bin_weights: list[float] = field(default_factory=lambda: [1.0])

    # cosine_sim params
    cosine_transform: str = "linear"   # "linear" | "softmax"
    cosine_temperature: float = 1.0

    # inv_rank params
    inv_rank_alpha: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "NegativeSelectionConfig":
        return cls(**d)


@dataclass
class ModelConfig:
    """Architecture config — two variants supported: 'mlp' and 'residual'."""
    arch: str                           # "mlp" | "residual"
    hidden_dims: list[int]              # e.g. [2048, 1024] or [1024, 1024, 1024]
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


@dataclass
class TrainConfig:
    dataset_dir: str                    # directory containing train.h5, val.h5, test.h5
    model: ModelConfig
    negative_selection: NegativeSelectionConfig
    temperature: float = 0.07
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    epochs: int = 50
    patience: int = 5                   # early stopping patience
    checkpoint_path: str = "checkpoints/contrastive_best.pt"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            dataset_dir=data["dataset_dir"],
            model=ModelConfig.from_dict(data["model"]),
            negative_selection=NegativeSelectionConfig.from_dict(data["negative_selection"]),
            temperature=data.get("temperature", 0.07),
            lr=data.get("lr", 1e-4),
            weight_decay=data.get("weight_decay", 1e-2),
            batch_size=data.get("batch_size", 256),
            epochs=data.get("epochs", 50),
            patience=data.get("patience", 5),
            checkpoint_path=data.get("checkpoint_path", "checkpoints/contrastive_best.pt"),
        )
