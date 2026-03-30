from abc import ABC, abstractmethod
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


# def embed_sentences(
#     df: pd.DataFrame, embedder: Embedder, target_column: str = "sent_no_cit", batch_size: int = 16
# ) -> pd.DataFrame:
#     tqdm.pandas(desc="Embedding sentences")
#     df["vector"] = df[target_column].progress_apply(lambda x: embedder([x])[0])
#     return df
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

    def __init__(self, db: MilvusDB, collection: str = "qwen06_chunks"):
        self.db = db
        self.collection = collection

    @abstractmethod
    def rank(self, queries: np.ndarray, candidates: np.ndarray, num: int) -> np.ndarray:
        """
        This ranks the search results (positive or negative) according to the class strategy,
        operating on the base numpy arrays
        """

    def rank_negatives(self, row: pd.DataFrame, mapped_queries: np.ndarray, num_negatives: int) -> np.ndarray:
        """
        Args:
            row: pd.DataFrame
                The dataframe row for the query, which should include:
                    - query_vectors: list[list[float]] one or more vector reps for this query
                    - pubdate: int YYYYMMDD used to filter search results to those published before the query document's pubdate
                    - citation_dois: the actual target documents for this query; used to filter out positives here
            num_negatives: int
                The number of negative examples to return per query
        """
        assert len(row) == len(mapped_queries), "Length of query dataframe must match length of mapped query array"
        results = []
        query_records = row.to_dict(orient="records")

        # For each record, search the db
        for record, query_reps in zip(query_records, mapped_queries):
            search_results = self.db.search(
                collection_name=self.collection,
                query_records=[record] * len(query_reps),  # Duplicate the record for each vector rep
                query_vectors=query_reps,
                limit=100,
                output_fields=["vector", "doi"],
            )[0]

            # Filter out any positive results and rank the negatives according to the strategy
            citation_dois = set(record["citation_dois"])
            negatives = np.array([res["vector"] for res in search_results if res["doi"] not in citation_dois])
            ranked_negatives = self.rank(query_reps, negatives, num_negatives)
            results.append(ranked_negatives)
        return np.array(results)

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
        """
        assert len(queries) == len(dois), "Number of query vector sets must match number of DOIs"
        results = []
        for query_reps, doi_list in zip(queries, dois):
            positives = []
            # For each target DOI, get its chunks from the db
            for doi in doi_list:
                chunks = DB.select_by_doi(doi=doi, collection_name=self.collection)
                vectors = np.array(chunks["vector"].tolist())
                ranked = self.rank(query_reps, vectors, num_positives)
                positives.append(ranked)
            results.append(np.array(positives))
            # positive_df = pd.concat(positive_chunks, ignore_index=True)

            # positive_vectors = np.array(positive_df["vector"].tolist())
            # ranked_positives = self.rank(query_reps, positive_vectors, num_positives)
            # results.append(ranked_positives)
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

        print(f"Shape of queries: {queries.shape}, Shape of candidates: {candidates.shape}")
        sort_indices = np.argsort(queries @ candidates.T)
        flat = sort_indices.flatten(order="F")
        interleaved_indices = np.array(list(dict.fromkeys(flat)))  # Preserve order while removing duplicates
        ranked_candidates = candidates[interleaved_indices]
        return ranked_candidates[:num]


class NNDatasetBuilder:
    def __init__(
        self,
        dataset: pd.DataFrame,
        strategy: MultiSimilarityStrategy,
        adapter: nn.Module,
        num_positives: int = 2,
        num_negatives: int = 4,
    ):
        self.dataset = dataset
        self.strategy = strategy
        self.adapter = adapter
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.mapped_queries = np.stack(
            [
                self.adapter(torch.tensor(np.stack(vecs)))
                for vecs in self.dataset["query_vectors"]
            ]
        )
        print(f"Mapped queries shape: {self.mapped_queries.shape}")

    def get_positives(self, df: pd.DataFrame, strategy: MultiSimilarityStrategy = None) -> np.ndarray:
        positives = strategy.rank_positives(queries=self.mapped_queries, dois=df["citation_dois"].tolist(), num_positives=self.num_positives)
        return positives

    def get_negatives(self, df: pd.DataFrame, strategy: MultiSimilarityStrategy = None) -> np.ndarray:
        negatives = strategy.rank_negatives(row=df, mapped_queries=self.mapped_queries, num_negatives=self.num_negatives)
        return negatives

    def build_and_save_h5(self, output_path: str):
        self.dataset["positives"] = self.get_positives(self.dataset, strategy=self.strategy).tolist()
        self.dataset["negatives"] = self.get_negatives(self.dataset, strategy=self.strategy).tolist()
        export_df = self.dataset[["query_vectors", "positives", "negatives"]].explode(["query_vectors"])
        queries = np.array(export_df["query_vectors"].tolist())
        positives = np.array(export_df["positives"].tolist())
        negatives = np.array(export_df["negatives"].tolist())
        with h5py.File(output_path, "w") as f:
            f.create_dataset("queries", data=queries)
            f.create_dataset("positives", data=positives)
            f.create_dataset("negatives", data=negatives)
        print(f"Dataset saved to: {output_path}")
        print(f"  Queries: {queries.shape}")
        print(f"  Positives: {positives.shape}")
        print(f"  Negatives: {negatives.shape}")


# def get_hard_negatives(
#     search_results: list[dict],
#     target_dois: set = None,
#     n: int = 3,
# ) -> list[str]:
#     """
#     Iterates over search results and returns a tensor of the first (hardest) n non-target vectors
#     """

#     vectors = []
#     for rec in search_results:
#         if rec["doi"] not in target_dois:
#             vectors.append(rec["vector"])
#         if len(vectors) == n:
#             break
#     else:
#         raise ValueError("Not enough hard negatives found in search results. Set top k higher?")

#     return torch.tensor(vectors)


# def get_soft_negatives(
#     search_results: list[dict],
#     target_dois: set = None,
#     n: int = 3,
# ) -> torch.tensor:
#     """
#     Assuming search_results > 100, samples n vectors from those results ranked higher than 100
#     """

#     assert len(search_results) > 100, "search_results must contain more than 100 entries"

#     # Filter down to only far results that are not targets
#     far_results = search_results[100:]
#     far_results = [rec for rec in far_results if rec["doi"] not in target_dois]
#     assert len(far_results) > n, "Not enough far results to sample from"

#     sampled_results = np.random.choice(far_results, size=n, replace=False)
#     vectors = [r["vector"] for r in sampled_results]
#     return torch.tensor(vectors)


# def get_hard_and_soft_negatives(
#     example: pd.Series,
#     hard_n: int = 3,
#     soft_n: int = 3,
#     db: MilvusDB = None,
#     collection: str = "qwen06_chunks",
#     top_k: int = 500,
# ) -> tuple[torch.tensor, torch.tensor]:
#     results = db.search(
#         collection_name=collection,
#         query_records=[example.to_dict()],
#         query_vectors=[example.vector],
#         limit=top_k,
#         output_fields=["text", "doi", "pubdate", "citation_count", "vector"],
#     )
#     results = results[0]  # db.search operates on lists of queries; we only need the first result
#     target_dois = set(example.citation_dois)  # All citation DOIs from original query (before explode)
#     hard_negatives = get_hard_negatives(results, target_dois=target_dois, n=hard_n)
#     soft_negatives = get_soft_negatives(results, target_dois=target_dois, n=soft_n)
#     return hard_negatives, soft_negatives


# def get_positives(example: pd.Series, n: int = 3, db: MilvusDB = None, collection: str = "qwen06_chunks") -> np.ndarray:
#     """
#     Returns n positive vectors from the single target DOI, ranked by similarity to the query
#     """
#     doi = example["target_doi"]  # Single target DOI after explode
#     records = db.select_by_doi(doi=doi, collection_name=collection)
#     target_vectors = np.array(records["vector"].tolist())

#     # Get the top n most similar records
#     similarities = example["vector"] @ target_vectors.T
#     top_n_indices = np.argpartition(-similarities, min(n, len(similarities)))[:n]
#     top_n_vectors = target_vectors[top_n_indices]
#     return top_n_vectors


# def append_query_expansion(df: pd.DataFrame, expander: str = "add_prev_2", batch_size: int = 32) -> pd.DataFrame:
#     """
#     Applies the query expansion function to all rows in the df, creating a new df denormalized on sent_no_cit
#     """
#     expander = QueryExpander(expander, reference_data_path="../../../data/preprocessed/reviews.jsonl")
#     df_expanded = df.copy()

#     # Apply query expansion in batches
#     for i in tqdm(range(0, len(df), batch_size), desc="Batch processing for query expansion"):
#         df_expanded.loc[i : i + batch_size, "sent_no_cit"] = expander(df_expanded.loc[i : i + batch_size])
#     df_out = pd.concat([df, df_expanded], ignore_index=True)
#     return df_out


# def data_to_h5(
#     df: pd.DataFrame,
#     output_path: str,
#     embedder: Embedder,
#     db: MilvusDB,
#     collection: str = "qwen06_chunks",
#     num_positives: int = 4,
#     num_hard_negatives: int = 4,
#     num_soft_negatives: int = 4,
# ):
#     df = embed_sentences(df, embedder)

#     queries = []
#     positives = []
#     hard_negatives = []
#     soft_negatives = []

#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
#         queries.append(row["vector"])
#         positives.append(get_positives(row, n=num_positives, db=db, collection=collection))
#         hard_negs, soft_negs = get_hard_and_soft_negatives(
#             row,
#             hard_n=num_hard_negatives,
#             soft_n=num_soft_negatives,
#             db=db,
#             collection=collection,
#         )
#         hard_negatives.append(hard_negs)
#         soft_negatives.append(soft_negs)

#     # Convert to numpy arrays
#     queries = np.array(queries)
#     positives = np.array(positives)
#     hard_negatives = np.array(hard_negatives)
#     soft_negatives = np.array(soft_negatives)

#     # Write to HDF5
#     with h5py.File(output_path, "w") as f:
#         f.create_dataset("queries", data=queries)
#         f.create_dataset("positives", data=positives)
#         f.create_dataset("hard_negatives", data=hard_negatives)
#         f.create_dataset("soft_negatives", data=soft_negatives)

#     print(f"Dataset saved to: {output_path}")
#     print(f"  Queries: {queries.shape}")
#     print(f"  Positives: {positives.shape}")
#     print(f"  Hard negatives: {hard_negatives.shape}")
#     print(f"  Soft negatives: {soft_negatives.shape}")


def main():
    """
    This script creates a dataset for neural net contrastive learning. The task is to map queries'
    embeddings close to their target documents' embeddings, and simultaneously far from hard and soft negatives.

    """
    DATA_FILE = "data/nn_datasets/val_dataset.parquet"
    OUTFILE_BASE = "data/nn_datasets"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    db = MilvusDB()
    df = pd.read_parquet(DATA_FILE)
    print(df.columns)
    print(f"Pubdate type: {type(df['pubdate'].iloc[0])}")

    interleaved_strategy = InterleavedStrategy(db=db, collection="qwen06_chunks")
    adapter = nn.Identity()
    builder = NNDatasetBuilder(
        dataset=df, 
        strategy=interleaved_strategy, 
        adapter=adapter, 
        num_positives=2, 
        num_negatives=4
    )
    output_path = f"{OUTFILE_BASE}/val_triplet_dataset_interleaved_strategy.h5"
    builder.build_and_save_h5(output_path)


if __name__ == "__main__":
    main()
