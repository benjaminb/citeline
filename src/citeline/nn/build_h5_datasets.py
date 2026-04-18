from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from itertools import product
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from citeline.database.milvusdb import MilvusDB
from citeline.embedders import Embedder
from citeline.query_expander import QueryExpander
from citeline.nn.config import H5DatasetWriterConfig
from citeline.nn.models import Adapter

DB = MilvusDB()
assert DB.healthcheck(), "MilvusDB healthcheck failed. Ensure Milvus server is running and accessible."

BATCH_SIZE = 8


class MultiSimilarityStrategy(ABC):
    """
    Abstract base class to implement a ranking strategy. The ranking strategy is implemented in rank() method,
    which takes the basic vectors (queries and candidates) as input and returns the ranked results.

    rank_positives and rank_negatives are implemented in the base class as their contract doesn't change; only
    the ranking strategy changes.

    This mostly comes into play when you have multiple query representatives.
    """

    registry = {}

    def __init__(self, collection: str = "qwen06_chunks"):
        self.collection = collection

    def __init_subclass__(cls, **kwargs):
        """Subclasses automatically register themselves in the registry dict"""
        super().__init_subclass__(**kwargs)
        MultiSimilarityStrategy.registry[cls.__name__] = cls

    @abstractmethod
    def rank(self, queries: np.ndarray, candidates: np.ndarray, num: int) -> np.ndarray:
        """
        This ranks the search results (positive or negative) according to the class strategy,
        operating on the base numpy arrays
        """

    def rank_negatives(
        self, row: pd.DataFrame, mapped_queries: np.ndarray, num_negatives: int, num_workers: int = 12
    ) -> list:
        """
        Args:
            row: pd.DataFrame
                The dataframe row for the query, which should include:
                    - query_vectors: list[list[float]] one or more vector reps for this query
                    - pubdate: int YYYYMMDD used to filter search results to those published before the query document's pubdate
                    - citation_dois: the actual target documents for this query; used to filter out positives here
            num_negatives: int
                The number of negative examples to return per query
            num_workers: int
                Number of threads for concurrent Milvus searches
        """
        assert len(row) == len(mapped_queries), "Length of query dataframe must match length of mapped query array"
        query_records = row.to_dict(orient="records")

        def search_one(args):
            thread_db = MilvusDB()  # Each thread needs its own connection to Milvus
            record, query_reps = args
            limit_pad = 10
            negatives = []
            while len(negatives) < num_negatives and limit_pad < 1000:
                search_results = thread_db.search(
                    collection_name=self.collection,
                    query_records=[record] * len(query_reps),  # Duplicate the record for each vector rep
                    query_vectors=query_reps,
                    limit=num_negatives + limit_pad,  # Retrieve more than needed to allow for filtering out positives
                    output_fields=["vector", "doi"],
                )[0]
                limit_pad *= 2  # Increase the limit_pad exponentially to avoid infinite loop

                citation_dois = set(record["citation_dois"])
                negatives = np.array([res["vector"] for res in search_results if res["doi"] not in citation_dois])
            return self.rank(query_reps, negatives, num_negatives)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(search_one, zip(query_records, mapped_queries)),
                    total=len(query_records),
                    desc="Ranking negatives",
                    leave=True,
                )
            )
        return results

    def rank_positives(self, queries: np.ndarray, dois: list[list[str]], num_positives: int) -> list:
        """
        Args:
            queries: np.ndarray
                The array of query vectors after mapping by the adapter net
                shape (Batch size x Number of Query Reps x Vector Dimension)
            dois: list[list[str]]
                The DOIs of the target documents
            num_positives: int
                The number of positive examples to return per query

        Returns:
            Returns a list of arrays with shape (num queries, num_positives, vector_dim)
        """
        assert len(queries) == len(dois), "Number of query vector sets must match number of DOIs"
        results = []
        for query_reps, doi_list in tqdm(zip(queries, dois), total=len(queries), desc="Ranking positives", leave=True):
            positives = []
            # For each target DOI, get its chunks from the db
            for doi in doi_list:
                chunks = DB.select_by_doi(doi=doi, collection_name=self.collection)
                vectors = np.array(chunks["vector"].tolist())
                # TODO: what happens if results < num_positives?
                ranked = self.rank(query_reps, vectors, num_positives)
                positives.append(ranked)
            results.append(np.array(positives))
        return results


class InterleavedStrategy(MultiSimilarityStrategy):
    def rank(self, queries: np.ndarray, candidates: np.ndarray, num: int) -> np.ndarray:
        """
        The point of view of this function is that `queries` is the array of query vector reps;
        so there are 1 or more of them (size num_reps x dim)

        Candidates is (N x dim) where N is the number of positive chunks for a given target DOI
        OR the number of negatives retrieved from the database
        """
        # Confirm you have enough vectors
        if len(candidates) < num:
            raise ValueError(f"Not enough candidate vectors. Required: {num}, Found: {len(candidates)}")

        sort_indices = np.argsort(queries @ candidates.T)
        flat = sort_indices.flatten(order="F")
        interleaved_indices = np.array(list(dict.fromkeys(flat)))  # Preserve order while removing duplicates
        ranked_candidates = candidates[interleaved_indices]
        return ranked_candidates[:num]


class H5DatasetWriter:
    def __init__(
        self,
        # Path to dir containing *train.parquet, *val.parquet, *test.parquet
        dataset_dir: str,
        strategy: MultiSimilarityStrategy,
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
        self, mapped_queries: np.ndarray, df: pd.DataFrame, strategy: MultiSimilarityStrategy = None
    ) -> np.ndarray:
        positives = strategy.rank_positives(
            queries=mapped_queries, dois=df["citation_dois"].tolist(), num_positives=self.num_positives
        )
        return positives

    def get_negatives(
        self, mapped_queries: np.ndarray, df: pd.DataFrame, strategy: MultiSimilarityStrategy = None
    ) -> np.ndarray:
        negatives = strategy.rank_negatives(row=df, mapped_queries=mapped_queries, num_negatives=self.num_negatives)
        return negatives

    def write_h5(self) -> None:
        # Guarantee the output path exists
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {}

        # Collect the input parquet files; confirme we have exactly the 3 we expect
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
                    np.pad(neg, ((0, max_negatives - len(neg)), (0, 0), (0, 0)), mode="constant")
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
    def from_config(cls, path: str) -> "H5DatasetWriter":
        config = H5DatasetWriterConfig.from_yaml(path)

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

        strategy_cls = MultiSimilarityStrategy.registry.get(config.strategy)
        if strategy_cls is None:
            raise ValueError(
                f"Strategy {config.strategy} not found in registry. Available strategies: {list(MultiSimilarityStrategy.registry.keys())}"
            )
        strategy = strategy_cls()

        adapter_cls = Adapter.registry.get(config.adapter)
        if adapter_cls is None:
            raise ValueError(
                f"Adapter {config.adapter} not found in registry. Available adapters: {list(Adapter.registry.keys())}"
            )
        adapter = adapter_cls()

        output_dir = Path(config.output_dir)
        return H5DatasetWriter(
            dataset_dir=dataset_dir,
            strategy=strategy,
            adapter=adapter,
            output_dir=output_dir,
            num_positives=config.num_positives,
            num_negatives=config.num_negatives,
        )


def main():
    """
    This script creates a dataset for neural net contrastive learning. The task is to map queries'
    embeddings close to their target documents' embeddings, and simultaneously far from hard and soft negatives.

    """
    DATA_FILE = "data/nn_datasets/val_dataset.parquet"


if __name__ == "__main__":
    main()
