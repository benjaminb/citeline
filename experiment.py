import argparse
import itertools
import json
import logging
import os
import torch
import yaml
import numpy as np
import pandas as pd

from time import time
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from pymilvus import Collection
from citeline.database.milvusdb import MilvusDB

# from citeline.query_expander import get_expander
from citeline.query_expander import QueryExpander
from citeline.embedders import Embedder

logger = logging.getLogger(__name__)
load_dotenv()

# Path to queries' full records, for query expansion
QUERY_EXPANSION_DATA = "data/preprocessed/reviews.jsonl"

"""
PATCH TO APPLY All-but-the-Top transformation to embeddings
"""
import pickle

pca = pickle.load(open("xtop_pca_1000.pkl", "rb"))
mean_vector = pickle.load(open("xtop_mean_vector_1000.pkl", "rb"))


def xtop_transform(vector: np.array, n: int) -> np.array:
    centered = vector - mean_vector
    projection = pca.components_ @ centered
    projection[:n] = 0  # Zero out the top-n components
    reconstructed = pca.components_.T @ projection

    norm = np.linalg.norm(reconstructed)
    if norm < 1e-10:
        print("Warning: Zero norm encountered in All-but-the-Top transformation.")
        return reconstructed
    return reconstructed / norm if norm else reconstructed


def argument_parser():
    """
    Example usage:
    python milvus_experiment.py --run experiments/configs/bert_cosine.yaml
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run an experiment with specified configuration or build a dataset.")

    # Create mutually exclusive operation groups
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        "--run",
        type=str,
        metavar="CONFIG_PATH",
        help="run an experiment with fixed top-k using the specified config file",
    )

    # Add a log level argument (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    return args


class Experiment:
    metric_to_str = {
        # PGVector metrics
        "vector_l2_ops": "L2",
        "vector_cosine_ops": "cosine",
        "vector_ip_ops": "ip",
        # Milvus metrics
        "L2": "l2",
        "IP": "ip",
        "COSINE": "cosine",
    }

    CLEAR_GPU_CACHE_FN = {"cuda": torch.cuda.empty_cache, "mps": torch.mps.empty_cache, "cpu": lambda: None}

    def __init__(
        self,
        **kwargs,
    ):
        """
        Args:
            kwargs: keyword arguments matching the parameters below
                dataset_path: path to a jsonl file containing lines that hat at least these keys:
                    "sent_no_cit": the sentence with inline citations replaced by "[REF]"
                    "sent_idx": the index of the sentence in the original record's "body_sentences" list
                    "source_doi": "...",
                    "
                    "pubdate": int (YYYYMMDD)
                - target_table: Name of the target table in the database.
                - target_column: Name of the target column.
                - metric: Metric to use for similarity (e.g., "cosine").
                - top_k: Number of top results to retrieve.
                - use_index: (postgres) Whether to use an index for the database.
                - probes: (postgres) Number of probes for the search.
                - ef_search: (postgres) ef_search parameter for the database.
                - distance_threshold: Distance threshold for filtering results.

                - embedding_model_name: Name of the embedding model.
                - normalize: Whether to normalize embeddings.

                - query_expansion: Query expansion strategy.
                - difference_vector_file: Path to the difference vector file.
                - transform_matrix_file: Path to the transform matrix file.
                - batch_size: Batch size for processing.

                - strategy: Search strategy to use.
                - query_expanders: List of query expanders to use (for multiple_query_expansion strategy).
                - interleave: Boolean for mixed search strategies. If False, sorts all search results by metric. If true, interleaves round-robin.
                - reranker_to_use: Reranker to use for results.
                - metrics_config: Configuration for metrics.
                - xtop: Use xtop transformation on embeddings.

                - output_path: Path to save results.
                - output_search_results: Whether to output ALL search results (top k per query).
        """
        self.config = kwargs

        # Dataset and results
        try:
            dataset_path = kwargs.get("dataset", None)
            self.dataset = pd.read_json(dataset_path, lines=True)
            self.dataset_path = dataset_path
        except Exception as e:
            raise ValueError(f"Error reading dataset from path '{dataset_path}': {e}")

        # Database parameters
        self.db = MilvusDB()
        self.top_k = kwargs.get("top_k", None)
        self.distance_threshold = kwargs.get("distance_threshold", None)

        # Postgres vector search parameters
        self.use_index = kwargs.get("use_index", True)
        self.probes = kwargs.get("probes", 16)
        self.ef_search = kwargs.get("ef_search", 40)

        # Querying parameters
        self.query_results: list[dict] = []
        self.first_rank: int | None = None  # The rank of the first target doi to appear in the results
        self.last_rank: int | None = None  # Rank at which all target DOIs appear in the results
        self.target_table = kwargs.get("table", None)
        self.target_column = kwargs.get("target_column", "vector")
        self.metric = kwargs.get("metric", "COSINE")

        # Embedding parameters
        self.batch_size = kwargs.get("batch_size", 16)
        self.normalize = kwargs.get("normalize", True)
        embedding_model_name = kwargs.get("embedder", None)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.embedder = Embedder.create(model_name=embedding_model_name, device=self.device, normalize=self.normalize)

        # Query expansion & linalg transformations
        query_expansion_name = kwargs.get("query_expansion", "identity")
        self.query_expander = (
            QueryExpander(query_expansion_name, reference_data_path=QUERY_EXPANSION_DATA)
            if query_expansion_name
            else None
        )
        self.reranker_to_use = kwargs.get("reranker_to_use", None)
        self.difference_vector_file = kwargs.get("difference_vector_file", None)
        self.difference_vector = np.load(self.difference_vector_file) if self.difference_vector_file else None
        self.transform_matrix_file = kwargs.get("transform_matrix_file", None)
        self.transform_matrix = np.load(self.transform_matrix_file) if self.transform_matrix_file else None
        self.metrics_config = kwargs.get("metrics_config", None)

        # Output parameters
        self.output_path = kwargs.get("output_path", None)
        self.output_search_results = kwargs.get("output_search_results", False)

        # Strategy for using multiple document / query expansions
        supported_strategies = {
            "50-50": self.__fifty_fifty_search,
            "basic": self.__basic_search,
            "mixed_expansion": self.__mixed_expansion_search,
            "multiple_query_expansion": self.__multiple_query_expansion_search,
        }

        strategy = kwargs.get("strategy", "basic")
        if strategy in supported_strategies:
            self.strategy = strategy
        else:
            raise ValueError(
                f"'{strategy}' is not a supported strategy. Supported strategies: {[k for k in supported_strategies.keys()]}"
            )
        self.search = supported_strategies[strategy]  # Search function to use in .run()
        query_expander_names = kwargs.get("query_expanders") if self.strategy == "multiple_query_expansion" else []
        self.query_expanders = (
            [QueryExpander(name, reference_data_path=QUERY_EXPANSION_DATA) for name in query_expander_names]
            if query_expander_names
            else None
        )
        self.interleave = kwargs.get("interleave", False) if self.query_expanders else False
        self.xtop = kwargs.get("xtop", False)
        self.xtop_n = kwargs.get("xtop_n", 0)  # Number of top components to remove (0 is falsey)

        # Prepare attributes for results
        self.recall_matrix = None
        self.hitrate_matrix = None
        self.iou_matrix = None
        self.avg_hitrate_at_k = None
        self.avg_iou_at_k = None
        self.avg_recall_at_k = None
        self.best_k_for_iou = None

        """
        Dictionary of average Jaccard scores for each distance threshold
        {0.5: 0.1785} means after only keeping query results with distance < 0.5, the average IoU score for
        all examples in the dataset is 0.1785
        """
        self.average_score = None  # For run method (no scan over top-k)
        self.average_scores = {}  # For run_and_topk_scan method (scan over top-k)

        # num_results tracks how many records the DB actually returned per query (can be less than top_k based on probes/ef_search)
        self.num_results = []
        self.best_top_ks = []
        self.best_top_distances = []
        self.hits = []
        self.recall_scores = []  # proportion of target DOIs that were retrieved in the top-k results

    def __basic_search(self, db: MilvusDB, records: list[dict], vectors: list[list[float]]) -> list[dict]:
        return db.search(
            collection_name=self.target_table,
            query_records=records,
            query_vectors=vectors,
            metric=self.metric,
            limit=self.top_k,
        )

    def __multiple_query_expansion_search(
        self,
        db: MilvusDB,
        records: pd.DataFrame,
        vectors: None,
        interleave=False,
    ) -> list[dict]:
        """
        This function handles multiple mixed query expansions (e.g. adding previous 1, 2, or 3 sentences) plus identity (no expansion)
        where each record has multiple associated vectors.

        Args:
            db: MilvusDB
            records: DataFrame with columns 'vector_original' and 'vector_{name of each expander}'
            vectors: None (not used, kept for compatibility)
            interleave: If True, interleaves results round-robin from each expansion. If False, sorts all results by metric.

        Note:
            - Always includes identity (no expansion) as one of the searches (I don't have a use case yet for not including it)
            - Expects list of 1 or more expanders in the attribute self.query_expanders (specified by name in YAML config)
        """
        assert (
            isinstance(self.query_expanders, list) and len(self.query_expanders) > 0
        ), "query_expanders must be a list of length ≥ 1 to use 'multiple query expansion' search strategy"
        num_expansions = len(self.query_expanders) + 1  # +1 for identity
        per_expansion_k = 1 + self.top_k // num_expansions

        batch_records = records.to_dict(orient="records")  # Convert once for db.search
        all_results = []

        # Do identity search first
        batch_vectors = records["vector_original"].tolist()
        results = db.search(
            collection_name=self.target_table,
            query_records=batch_records,
            query_vectors=batch_vectors,
            metric=self.metric,
            limit=per_expansion_k,
        )
        all_results.append(results)

        # Do expanded query searches (use DataFrame 'records' for column access)
        for expander in self.query_expanders:
            # Requires that the main thread in run() puts embeddings on the df using colnames "vector_{name of expander}
            batch_vectors = records[f"vector_{expander.name}"].tolist()
            results = db.search(
                collection_name=self.target_table,
                query_records=batch_records,
                query_vectors=batch_vectors,
                metric=self.metric,
                limit=per_expansion_k,
            )
            all_results.append(results)

        # all_results is now a list of 4 search results (each is list of length batch_size)
        # For each query index, interleave or merge the 4 expansion results
        batch_size = len(records)
        combined_results = []

        for i in range(batch_size):
            # Gather the i-th query's results from each expansion
            per_expansion = [all_results[exp][i] if i < len(all_results[exp]) else [] for exp in range(num_expansions)]

            if interleave:
                # Interleave round-robin across expansions
                merged = []
                max_len = max((len(lst) for lst in per_expansion), default=0)
                for j in range(max_len):
                    for lst in per_expansion:
                        if j < len(lst) and lst[j] is not None:
                            merged.append(lst[j])
                merged = merged[: self.top_k]
            else:
                # Combine all and sort by metric
                merged = []
                for lst in per_expansion:
                    merged.extend([x for x in lst if x is not None])
                merged.sort(key=lambda x: x["metric"], reverse=True)
                merged = merged[: self.top_k]

            combined_results.append(merged)

        return combined_results

    def __mixed_expansion_search(self, db: MilvusDB, records: pd.DataFrame, vectors: None) -> list[dict]:
        """
        This function handles the "mixed_expansion" strategy where each record has two associated vectors:
        the original and the expanded. It retrieves top-k/2 results for each vector and combines them.

        Expects records to be a DataFrame with 'vector_original' and 'vector_expanded' columns
        """
        half_k = self.top_k // 2

        original_vectors = records["vector_original"].tolist()
        expanded_vectors = records["vector_expanded"].tolist()
        batch_records = records.to_dict(orient="records")

        original_results = db.search(
            collection_name=self.target_table,
            query_records=batch_records,
            query_vectors=original_vectors,
            metric=self.metric,
            limit=half_k,
        )

        expanded_results = db.search(
            collection_name=self.target_table,
            query_records=batch_records,
            query_vectors=expanded_vectors,
            metric=self.metric,
            limit=half_k,
        )

        # Combine search results and sort by metric
        results = []
        for i in range(len(records)):
            original_search_results = original_results[i]
            expanded_search_results = expanded_results[i]
            combined_results = original_search_results + expanded_search_results
            combined_results.sort(key=lambda x: x["metric"], reverse=True)  # Sorts by metric DESCENDING
            results.append(combined_results)
        return results

    def __fifty_fifty_search(self, db: MilvusDB, records: list[dict], vectors: list[list[float]]) -> list[dict]:
        """
        This function retrieves half the top-k from chunks, and half from contributions,
        then returns the interleaved results

        Expects self.table_name to be a stub such as "bge_" to which we concatenate "chunks"
        and "contributions" and so on
        """
        half_k = self.top_k // 2

        chunk_results = db.search(
            collection_name=self.target_table + "chunks",
            query_records=records,
            query_vectors=vectors,
            metric=self.metric,
            limit=half_k,
        )

        contribution_results = db.search(
            collection_name=self.target_table + "contributions",
            query_records=records,
            query_vectors=vectors,
            metric=self.metric,
            limit=half_k,
        )

        # Combine search results
        results = []
        for i in range(len(records)):  # batch size
            chunk_search_results = chunk_results[i]
            contribution_search_results = contribution_results[i]
            interleaved_results = []
            for j in range(half_k):
                interleaved_results.append(chunk_search_results[j])
                interleaved_results.append(contribution_search_results[j])
            results.append(interleaved_results)
        return results

    def _evaluate_prediction(self, example, results):
        unique_predicted_dois = set(result.doi for result in results)
        citation_dois = set(doi for doi in example.citation_dois)
        score = self.__jaccard_score(unique_predicted_dois, citation_dois)
        return score

    def _evaluate_prediction_dict(self, example, results):
        unique_predicted_dois = set(result.doi for result in results)
        citation_dois = set(doi for doi in example["citation_dois"])
        score = self.__jaccard_score(unique_predicted_dois, citation_dois)
        return score

    def __jaccard_score(self, set1, set2):
        intersection = np.longdouble(len(set1.intersection(set2)))
        union = np.longdouble(len(set1.union(set2)))
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _get_output_filename_base(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if label := self.config.get("plot_label"):
            label = label.replace("/", "_")
            return f"{label}_{current_time}"

        reranker_str = f"_{self.reranker_to_use}" if self.reranker_to_use else ""
        diff = "_diff" if self.difference_vector_file else ""

        if self.query_expander:  # Using a single query expander
            query_expansion_str = "_" + self.query_expander.name
        else:  # Using multiple query expanders (reflected in self.strategy)
            query_expansion_str = ""
        return f"{self.strategy}_{self.target_table}{query_expansion_str}{diff}{reranker_str}_norm{self.normalize}_{self.metric_to_str[self.metric]}_n{len(self.dataset)}_{current_time}"

    def __write_run_results(self):
        """
        Writes out the results of a .run() experiment, which only includes the config and the average Jaccard score.
        """
        filename_base = self._get_output_filename_base()

        # Create directory if it doesn't exist
        if not os.path.exists(f"{self.output_path}/{filename_base}"):
            os.makedirs(f"{self.output_path}/{filename_base}")

        output = {
            "config": self.config,
            "average_score": self.average_score,
            "ef_search": self.ef_search,
            "average_hitrate_at_k": self.avg_hitrate_at_k,
            "average_iou_at_k": self.avg_iou_at_k,
            "average_recall_at_k": self.avg_recall_at_k,
            "best_recall_k": max(self.avg_recall_at_k),
            "best_iou_k": max(self.avg_iou_at_k),
        }

        file_path = os.path.join(self.output_path, filename_base, f"results_{filename_base}.json")
        print(f"Writing output to {self.output_path}{filename_base}")
        with open(file_path, "w") as f:
            json.dump(output, f)

        self.__plot_results(filename_base)

    def __plot_results(self, filename_base):
        import matplotlib.pyplot as plt

        k_values = [k for k in range(1, self.top_k + 1)]
        plt.figure(figsize=(12, 8))  # Slightly larger to accommodate annotations

        # Plot the lines
        (line1,) = plt.plot(k_values, self.avg_hitrate_at_k, linestyle="-", label="Average Hit Rate@k", color="blue")
        (line2,) = plt.plot(k_values, self.avg_iou_at_k, linestyle="-", label="Average IoU@k", color="green")
        (line3,) = plt.plot(k_values, self.avg_recall_at_k, linestyle="-", label="Average Recall@k", color="red")

        # Add marker and label for maximal IoU point
        max_iou_value = self.avg_iou_at_k[self.best_k_for_iou - 1]
        plt.scatter(self.best_k_for_iou, max(self.avg_iou_at_k), color="green", s=100, zorder=5, marker="o")
        plt.annotate(
            f"Max IoU: {max_iou_value:.3f} at k={self.best_k_for_iou}",
            xy=(self.best_k_for_iou, max_iou_value),
            xytext=(20, 20),
            textcoords="offset points",
            fontsize=10,
            color="green",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
        )

        # Add annotations every 100 values
        annotation_interval = 100
        for i in range(0, len(k_values), annotation_interval):
            k = k_values[i]

            # Annotate hit rate
            plt.annotate(
                f"{self.avg_hitrate_at_k[i]:.3f}",
                xy=(k, self.avg_hitrate_at_k[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="blue",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

            # Annotate IoU
            plt.annotate(
                f"{self.avg_iou_at_k[i]:.3f}",
                xy=(k, self.avg_iou_at_k[i]),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=8,
                color="green",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

            # Annotate recall
            plt.annotate(
                f"{self.avg_recall_at_k[i]:.3f}",
                xy=(k, self.avg_recall_at_k[i]),
                xytext=(5, -25),
                textcoords="offset points",
                fontsize=8,
                color="red",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        plt.xlabel("Top-k")
        plt.ylabel("Score")
        plt.title("Stats@k")
        plt.legend()

        # Add grid lines at 0.05 intervals but labels every 0.1
        from matplotlib.ticker import MultipleLocator

        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Labels every 0.1
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))  # Grid lines every 0.05
        plt.grid(True, alpha=0.3, which="both")  # Show both major and minor grid lines

        plt.tight_layout()  # Adjust layout to prevent clipping of annotations
        plt.savefig(f"{self.output_path}/{filename_base}/stats_at_k_{filename_base}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def __clear_gpu_cache(self):
        self.CLEAR_GPU_CACHE_FN[self.device]()

    def run(self):
        """
        Run the experiment with the producer on the main thread and consumer threads:
        - Producer (embedding generation) runs on the main thread with GPU
        - Multiple consumer threads for database operations

        This function doesn't compute metrics over various top-k cutoffs; just given the
        experiment config with a set top k
        """

        from concurrent.futures import ThreadPoolExecutor
        import queue
        import threading

        # Make sure we have CPU count available
        assert "CPUS" in os.environ, "CPUS environment variable not set."
        try:
            num_cpus = int(os.getenv("CPUS"))
        except ValueError:
            raise ValueError(f"Invalid value for CPUS environment variable.")
        if self.strategy != "50-50":
            collection = Collection(name=self.target_table)
            collection.load()
            print(f"Collection {self.target_table} loaded.")

        # Create thread-safe queues for tasks and results
        task_queue = queue.Queue(maxsize=128)
        progress_bar_lock = threading.Lock()
        stats_lock = threading.Lock()
        file_lock = threading.Lock()
        # results_queue = queue.Queue()

        # Preallocate result matrices so consumers can write into them directly
        dataset_size = len(self.dataset)
        self.hitrate_matrix = np.zeros((dataset_size, self.top_k))
        self.iou_matrix = np.zeros((dataset_size, self.top_k))
        self.recall_matrix = np.zeros((dataset_size, self.top_k))

        consumer_progress = 0

        consumer_bar = None  # Declare consumer bar reference; initialized during producer startup
        sentinel = object()  # Unique sentinel object for signaling completion

        # Prepare output file if streaming search results
        out_file = None
        if self.output_search_results:
            try:
                os.makedirs(self.output_path, exist_ok=True)
                out_path = os.path.join(self.output_path, "search_results.jsonl")
                out_file = open(out_path, "w", encoding="utf-8")
                print(f"Streaming search results to {out_path}")
            except Exception as e:
                logger.error(f"Could not open search results file for writing: {e}")
                out_file = None

        def consumer_thread():
            """Consumer thread that handles database queries and evaluations"""
            nonlocal consumer_progress
            thread_client = MilvusDB()

            while True:
                item = task_queue.get()
                try:
                    if item is sentinel:
                        break

                    # Query the database
                    batch, embeddings, start_idx = item

                    # TODO: simplify this logic!!!
                    if self.xtop and embeddings is not None:
                        # Apply All-but-the-Top transformation to each embedding
                        embeddings = [xtop_transform(np.array(vec), self.xtop_n).tolist() for vec in embeddings]
                    elif self.xtop and self.strategy in ["mixed_expansion", "multiple_query_expansion"]:
                        # For strategies that store vectors in the batch DataFrame, apply xtop to those
                        if "vector_original" in batch.columns:
                            batch["vector_original"] = [
                                xtop_transform(np.array(vec), self.xtop_n).tolist() for vec in batch["vector_original"]
                            ]
                        if self.strategy == "multiple_query_expansion" and self.query_expanders:
                            for expander in self.query_expanders:
                                col_name = f"vector_{expander.name}"
                                if col_name in batch.columns:
                                    batch[col_name] = [
                                        xtop_transform(np.array(vec), self.xtop_n).tolist() for vec in batch[col_name]
                                    ]
                    if self.strategy in ["mixed_expansion"]:
                        # TODO: this is just a patch for mixed expansion search. If we want to keep this strategy, come up with a cleaner design
                        embeddings = batch["vector_original"].tolist()  # Dummy, not used
                        search_results = self.search(db=thread_client, records=batch, vectors=None)
                        batch_records = batch.to_dict(orient="records")
                    elif self.strategy == "multiple_query_expansion":
                        # TODO: this is just a patch for mixed expansion search. If we want to keep this strategy, come up with a cleaner design
                        embeddings = batch["vector_original"].tolist()  # Dummy, not used
                        search_results = self.search(
                            db=thread_client, records=batch, vectors=None, interleave=self.interleave
                        )
                        batch_records = batch.to_dict(orient="records")
                    else:
                        batch_records = batch.to_dict(orient="records")
                        search_results = self.search(db=thread_client, records=batch_records, vectors=embeddings)

                    # TODO: fix logging within thread

                    # Log any anomalies in record retrieval
                    if len(search_results) != len(embeddings):
                        logger.warning(f"Expected {len(embeddings)} results, but got {len(search_results)} for batch")
                        print(
                            f"Expected {len(embeddings)} results, but got {len(search_results)} for batch", flush=True
                        )
                    if len(search_results[0]) != self.top_k:
                        logger.warning(f"Expected {self.top_k} results, but got {len(search_results[0])} for batch.")
                        print(f"Expected {self.top_k} results, but got {len(search_results[0])} for batch.", flush=True)

                    # Compute batch metrics locally
                    stats = self._compute_metrics_batch(batch_records, search_results)

                    # Write into global matrices under lock using the known start index
                    with stats_lock:
                        end_idx = start_idx + len(batch_records)
                        self.recall_matrix[start_idx:end_idx, :] = stats["recall_at_k"]
                        self.iou_matrix[start_idx:end_idx, :] = stats["iou_at_k"]
                        self.hitrate_matrix[start_idx:end_idx, :] = stats["hitrate_at_k"]

                    # Optionally stream individual results to disk safely
                    if self.output_search_results and out_file is not None:
                        with file_lock:
                            for rec, res in zip(batch_records, search_results):
                                try:
                                    out_file.write(
                                        json.dumps({"record": rec, "results": res}, ensure_ascii=False) + "\n"
                                    )
                                except Exception as e:
                                    logger.error(f"Failed writing a search-result line: {e}")

                    # Update DB-queries progress bar
                    with progress_bar_lock:
                        consumer_bar.update(len(embeddings))

                except Exception as e:
                    print(f"Consumer thread error: {str(e)}")
                finally:
                    task_queue.task_done()

        # Start consumer threads
        num_workers = max(1, num_cpus - 1)  # Leave one core for the main thread

        # if self.reranker_to_use == "deberta_nli":
        #     num_workers = min(num_workers, 6)  # Limit to 6 workers for DeBERTa reranker, which must host another model and process on GPU
        print(f"Starting {num_workers} database query workers")

        start = time()
        consumer_threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create progress bars
            with tqdm(total=dataset_size, desc="Embedding (GPU)", position=0) as producer_bar, tqdm(
                total=dataset_size, desc="DB Queries", position=1
            ) as cbar:
                # Initialize consumer bar ref, must be before consumers start
                consumer_bar = cbar

                # Start consumer threads
                for _ in range(num_workers):
                    thread = executor.submit(consumer_thread)
                    consumer_threads.append(thread)

                # Main thread acts as the producer, clear GPU cache every 50 batches
                clear_cache_interval = self.batch_size * 50
                for i in range(0, dataset_size, self.batch_size):
                    if i % clear_cache_interval == 0:
                        self.__clear_gpu_cache()

                    # Get batch, perform any query expansion & generate embeddings
                    # Avoid modifying original dataset
                    batch = self.dataset.iloc[slice(i, i + self.batch_size)].copy()
                    if self.strategy == "mixed_expansion":
                        """
                        Hacky patch to handle mixed expansion strategy where we need to send two reps per query
                        into the task queue
                        """
                        batch["expansion"] = self.query_expander(batch)
                        batch["vector_original"] = [vector for vector in self.embedder(batch["sent_no_cit"])]
                        batch["vector_expanded"] = [vector for vector in self.embedder(batch["expansion"])]
                        task_queue.put((batch, None, i))
                    elif self.strategy == "multiple_query_expansion":
                        batch = batch.copy()
                        # Embed the original (identity) expansion
                        batch["vector_original"] = [vector for vector in self.embedder(batch["sent_no_cit"])]
                        # Embed the "add previous n sentences" expansions
                        for idx, expander in enumerate(self.query_expanders):
                            expansions = expander(batch)
                            expansion_name = expander.name
                            batch[expansion_name] = expansions
                            batch[f"vector_{expansion_name}"] = [vector for vector in self.embedder(expansions)]
                        task_queue.put((batch, None, i))
                    else:
                        expanded_queries = self.query_expander(batch)
                        embeddings = self.embedder(expanded_queries)
                        # Apply any vector transformation if specified
                        if self.difference_vector is not None:
                            embeddings = embeddings + self.difference_vector
                            # Renormalize if needed
                            if self.embedder.normalize:
                                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                                embeddings = embeddings / norms
                        elif self.transform_matrix is not None:
                            embeddings = embeddings @ self.transform_matrix
                            # Renormalize if needed
                            if self.embedder.normalize:
                                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                                embeddings = embeddings / norms

                        task_queue.put((batch, embeddings.tolist(), i))

                    producer_bar.update(len(batch))

                # Put sentinels on the task queue to signal consumer completion
                for _ in range(num_workers):
                    task_queue.put(sentinel)

                # Cleanup the producer
                task_queue.join()

        # Close the output file if used
        if out_file is not None:
            try:
                out_file.close()
            except Exception:
                pass

        print(f"Experiment computed in {time() - start:.2f} seconds")

        # Ensure we processed everything we enqueued
        # if stats_idx != len(self.dataset):
        #     logger.warning(f"Stats rows filled ({stats_idx}) != dataset size ({len(self.dataset)}).")

        # Compute summary stats
        self.avg_recall_at_k = self.recall_matrix.mean(axis=0).tolist()
        self.avg_hitrate_at_k = self.hitrate_matrix.mean(axis=0).tolist()
        self.avg_iou_at_k = self.iou_matrix.mean(axis=0).tolist()
        self.best_k_for_iou = int(np.argmax(self.avg_iou_at_k)) + 1  # +1 for 1-indexed k
        self.__write_run_results()

    def __str__(self):
        s = "=" * 20 + " Experiment Configuration " + "=" * 20 + "\n"
        for key, value in self.config.items():
            s += f"{key}: {value}\n"
        s += "=" * 60 + "\n"
        return s

    def _compute_metrics(self, example, results) -> dict[str, np.ndarray]:
        """
        Computes the metrics for a single example and its results.
        Returns a dictionary of metrics.
        """
        hit_dois = set()  # Dois in retrieved entities that match a target doi
        target_dois = example["citation_dois"]
        retrieved_dois = set()
        union_dois = set(target_dois)
        hitrate_at_k = np.zeros(self.top_k)
        iou_at_k = np.zeros(self.top_k)
        recall_at_k = np.zeros(self.top_k)

        # Iterate over the results
        for i, result in enumerate(results):
            # Add retrieved DOI at this rank to retrieved_dois and union
            doi = result["doi"]
            retrieved_dois.add(doi)
            union_dois.add(doi)
            if doi in target_dois:
                hit_dois.add(doi)

            # Compute and store stats
            recall = len(hit_dois) / len(target_dois) if target_dois else 0
            hitrate = int(recall > 0)  # 1 if hit, 0 otherwise
            iou = len(hit_dois) / len(union_dois) if union_dois else 0

            recall_at_k[i] = recall
            hitrate_at_k[i] = hitrate
            iou_at_k[i] = iou

        return {
            "recall_at_k": recall_at_k,
            "hitrate_at_k": hitrate_at_k,
            "iou_at_k": iou_at_k,
        }

    def _compute_metrics_batch(self, examples, batch_results):

        # Initialize metrics
        batch_recall = np.zeros((len(examples), self.top_k))
        batch_hitrate = np.zeros((len(examples), self.top_k))
        batch_iou = np.zeros((len(examples), self.top_k))

        # Compute metrics for each example in the batch
        for i, (example, results) in enumerate(zip(examples, batch_results)):
            metrics = self._compute_metrics(example, results)
            batch_recall[i] = metrics["recall_at_k"]
            batch_hitrate[i] = metrics["hitrate_at_k"]
            batch_iou[i] = metrics["iou_at_k"]

        # Aggregate metrics across the batch
        return {
            "recall_at_k": batch_recall,
            "hitrate_at_k": batch_hitrate,
            "iou_at_k": batch_iou,
        }


def main():

    args = argument_parser()

    # Set up logging
    logging.basicConfig(
        filename="logs/experiment.log",
        filemode="w",
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.run:
        # Load experiment configs
        config_filename = args.run
        with open(config_filename, "r") as config_file:
            config = yaml.safe_load(config_file)
            config["config_file"] = config_filename  # Add config filename to config dict

        experiment = Experiment(**config)
        print(experiment)
        experiment.run()

        return

    if args.write:
        train, test = train_test_split_nontrivial("data/dataset/full/nontrivial.jsonl")
        write_train_test_to_file(train, test, "data/dataset/split/")
        return


if __name__ == "__main__":
    main()
