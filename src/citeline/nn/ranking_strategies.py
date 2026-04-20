from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from citeline.database.milvusdb import MilvusDB


class RankingStrategy(ABC):
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
        self.db = MilvusDB()

    def __init_subclass__(cls, **kwargs):
        """Subclasses automatically register themselves in the registry dict"""
        super().__init_subclass__(**kwargs)
        RankingStrategy.registry[cls.__name__] = cls

    @abstractmethod
    def rank(self, queries: np.ndarray, candidates: np.ndarray, num: int) -> np.ndarray:
        """
        This ranks the search results (positive or negative) according to the class strategy,
        operating on the base numpy arrays
        """

    def rank_negatives(
        self, row: pd.DataFrame, mapped_queries: np.ndarray, num_negatives: int, num_workers: int = 12, nprobe: int = 64
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
            nprobe: int
                Number of clusters to probe for IVF_FLAT index. Higher = more accurate, slower.
                Pass None to omit (required for FLAT index).
        """
        assert len(row) == len(mapped_queries), "Length of query dataframe must match length of mapped query array"
        query_records = row.to_dict(orient="records")
        _thread_local = threading.local()

        def search_one(args):
            # Reuse one connection per thread instead of creating a new one per task
            if not hasattr(_thread_local, "db"):
                _thread_local.db = MilvusDB()
            thread_db = _thread_local.db

            record, query_reps = args
            citation_dois = set(record["citation_dois"])
            limit_pad = num_negatives * 2
            negatives = []
            while len(negatives) < num_negatives and limit_pad < 10000:
                all_results = thread_db.search(
                    collection_name=self.collection,
                    query_records=[record] * len(query_reps),
                    query_vectors=query_reps,
                    limit=num_negatives + limit_pad,
                    output_fields=["vector", "doi"],
                    nprobe=nprobe,
                )
                limit_pad *= 2  # Increase the limit_pad exponentially to avoid infinite loop

                # Merge results from all query reps, deduplicating by DOI
                seen_dois = set()
                merged = []
                for result_list in all_results:
                    for res in result_list:
                        if res["doi"] not in seen_dois:
                            seen_dois.add(res["doi"])
                            merged.append(res)

                negatives = np.array([res["vector"] for res in merged if res["doi"] not in citation_dois])
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
                chunks = self.db.select_by_doi(doi=doi, collection_name=self.collection)
                vectors = np.array(chunks["vector"].tolist())
                # TODO: what happens if results < num_positives?
                ranked = self.rank(query_reps, vectors, num_positives)
                positives.append(ranked)
            results.append(np.array(positives))
        return results


class InterleavedStrategy(RankingStrategy):
    def rank(self, queries: np.ndarray, candidates: np.ndarray, num: int) -> np.ndarray:
        """
        The point of view of this function is that `queries` is the array of query vector reps;
        so there are 1 or more of them (size num_reps x dim)

        Candidates is (N x dim) where N is the number of positive chunks for a given target DOI
        OR the number of negatives retrieved from the database

        For each query rep, sorts candidates by descending similarity, then interleaves the ranked
        lists across reps (column-major flatten) so the top results alternate between reps.
        Duplicates are removed while preserving order.
        """
        # Confirm you have enough vectors
        if len(candidates) < num:
            raise ValueError(f"Not enough candidate vectors. Required: {num}, Found: {len(candidates)}")

        sort_indices = np.argsort(-(queries @ candidates.T))
        flat = sort_indices.flatten(order="F")
        interleaved_indices = np.array(list(dict.fromkeys(flat)))  # Preserve order while removing duplicates
        ranked_candidates = candidates[interleaved_indices]
        return ranked_candidates[:num]
