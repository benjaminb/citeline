from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from citeline.nn.config import ContrastiveDatasetBuilderConfig
from citeline.nn.models import Adapter
from citeline.nn.ranking_strategies import RankingStrategy


class ContrastiveDatasetWriter:
    def __init__(
        self,
        # Path to dir containing *train.parquet, *val.parquet, *test.parquet
        dataset_dir: str,
        strategy: RankingStrategy,
        adapter: nn.Module,
        output_dir: str,
        num_positives: int = 2,
        num_negatives: int = 4,
    ):
        self.dataset_dir = dataset_dir
        self.strategy = strategy
        self.adapter = adapter
        self.device = next(adapter.parameters()).device
        self.output_dir = output_dir
        self.num_positives = num_positives
        self.num_negatives = num_negatives

    def get_positives(
        self, mapped_queries: np.ndarray, df: pd.DataFrame, strategy: RankingStrategy = None
    ) -> np.ndarray:
        positives = strategy.rank_positives(
            queries=mapped_queries, dois=df["citation_dois"].tolist(), num_positives=self.num_positives
        )
        return positives

    def get_negatives(
        self, mapped_queries: np.ndarray, df: pd.DataFrame, strategy: RankingStrategy = None
    ) -> np.ndarray:
        negatives = strategy.rank_negatives(row=df, mapped_queries=mapped_queries, num_negatives=self.num_negatives)
        return negatives

    def write_h5(self) -> None:
        # Guarantee the output path exists
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {}

        # Collect the input parquet files; confirm we have exactly the 3 we expect
        files = list(Path(self.dataset_dir).glob("*.parquet"))
        for parquet_path in files:
            print(f"Processing {parquet_path.stem}...")
            dataset = pd.read_parquet(parquet_path)
            mapped_queries = np.stack(
                [
                    self.adapter(torch.from_numpy(np.stack(vecs).astype(np.float32)).to(self.device))
                    .cpu()
                    .detach()
                    .numpy()
                    for vecs in dataset["query_vectors"]
                ]
            )

            dataset["positives"] = self.get_positives(mapped_queries, dataset, strategy=self.strategy)
            dataset["num_targets"] = np.array([len(pos_list) for pos_list in dataset["positives"]])
            dataset["negatives"] = self.get_negatives(mapped_queries, dataset, strategy=self.strategy)

            # This explode causes us to have one anchor per query; multi-anchor approaches outside this project's scope
            export_df = dataset[["query_vectors", "positives", "negatives", "num_targets"]].explode(["query_vectors"])

            # Pad the positives and negatives to ensure they have consistent shapes for saving to h5
            print("Padding shapes...", end="\r", flush=True)
            max_positives = max(len(pos) for pos in export_df["positives"])
            max_negatives = max(len(neg) for neg in export_df["negatives"])
            export_df["positives"] = export_df["positives"].apply(
                lambda pos: (
                    np.pad(pos, ((0, max_positives - len(pos)), (0, 0), (0, 0)), mode="constant")
                    if len(pos) < max_positives
                    else pos
                )
            )
            export_df["negatives"] = export_df["negatives"].apply(
                lambda neg: (
                    np.pad(neg, ((0, max_negatives - len(neg)), (0, 0)), mode="constant")
                    if len(neg) < max_negatives
                    else neg
                )
            )

            # Convert pd.Series -> np.ndarray for h5 write
            anchors = np.stack(export_df["query_vectors"].tolist())
            num_positives = export_df["num_targets"].to_numpy()
            positives = np.stack(export_df["positives"].tolist())
            negatives = np.stack(export_df["negatives"].tolist())

            # Resolve output filename and path; write the h5 file
            outfile_name = parquet_path.name.replace(".parquet", ".h5")
            output_path = Path(self.output_dir) / outfile_name
            with h5py.File(output_path, "w") as f:
                f.create_dataset("queries", data=anchors)
                f.create_dataset("num_targets", data=num_positives)
                f.create_dataset("positives", data=positives)
                f.create_dataset("negatives", data=negatives)
            print(f"Dataset saved to: {output_path}")
            print(f"  Queries: {anchors.shape}")
            print(f"  Positives: {positives.shape}")
            print(f"  Negatives: {negatives.shape}")
            output_paths[parquet_path.stem] = output_path
        return output_paths

    @classmethod
    def from_config(cls, path: str) -> "ContrastiveDatasetWriter":
        config = ContrastiveDatasetBuilderConfig.from_yaml(path)

        dataset_dir = Path(config.dataset_dir)
        parquet_files = list(dataset_dir.glob("*.parquet"))
        assert (
            len(parquet_files) == 3
        ), f"Expected exactly 3 .parquet files in {dataset_dir}, found {len(parquet_files)}"

        splits = {}
        for split in ("train", "val", "test"):
            matches = [p for p in parquet_files if p.stem.endswith(split)]
            assert len(matches) == 1, f"Expected exactly one *{split}.parquet in {dataset_dir}, found {len(matches)}"
            splits[split] = matches[0]

        strategy_cls = RankingStrategy.registry.get(config.strategy)
        if strategy_cls is None:
            raise ValueError(
                f"Strategy {config.strategy} not found in registry. Available strategies: {list(RankingStrategy.registry.keys())}"
            )
        strategy = strategy_cls()

        adapter_cls = Adapter.registry.get(config.adapter)
        if adapter_cls is None:
            raise ValueError(
                f"Adapter {config.adapter} not found in registry. Available adapters: {list(Adapter.registry.keys())}"
            )
        adapter = adapter_cls()

        output_dir = Path(config.output_dir)
        return ContrastiveDatasetWriter(
            dataset_dir=dataset_dir,
            strategy=strategy,
            adapter=adapter,
            output_dir=output_dir,
            num_positives=config.num_positives,
            num_negatives=config.num_negatives,
        )
