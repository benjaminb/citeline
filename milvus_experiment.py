import argparse
import json
import logging
import os
import torch
import yaml
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from pymilvus import Collection
from database.milvusdb import MilvusDB
from query_expander import get_expander
from embedders import Embedder

# from rerankers import get_reranker
# from metrics import RankFuser

logger = logging.getLogger(__name__)
load_dotenv()
# DISTANCE_THRESHOLDS = np.arange(1.0, 0.0, -0.01)

EXPANSION_DATA_PATH = "data/preprocessed/reviews.jsonl"


def argument_parser():
    """
    Example usage:

    1. Run an experiment with specified configuration:
       python milvus_experiment.py --run experiments/configs/bert_cosine.yaml

    2. Build a train/test split by sampling from source:
       python milvus_experiment.py --build --source data/dataset/full/nontrivial_llm.jsonl --train-dest data/dataset/sampled/train.jsonl --test-dest data/dataset/sample/test.jsonl --split=0.8 --seed 42

    3. Run an experiment with a top-k scan:
       python milvus_experiment.py --run-scan experiments/bert_cosine.yaml

    4. Generate query plans and analyze database performance:
       python milvus_experiment.py --query-plan --table-name bert_hnsw --embedder bert-base-uncased --top-k 50

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
    operation_group.add_argument(
        "--run-scan",
        type=str,
        metavar="CONFIG_PATH",
        help="run an experiment with top-k scan using the specified config file",
    )
    operation_group.add_argument("--build", action="store_true", help="build a dataset")
    operation_group.add_argument("--write", action="store_true", help="write out train/test datasets")
    operation_group.add_argument(
        "--query-plan", action="store_true", help="generate EXPLAIN/ANALYZE query plan for database"
    )

    # Dataset building arguments
    parser.add_argument("--num", type=int, help="number of examples to include")
    parser.add_argument("--source", type=str, help="path to source dataset (jsonl)")
    parser.add_argument("--train-dest", type=str, help="save path for training set (jsonl)")
    parser.add_argument("--test-dest", type=str, help="save path for test set (jsonl)")
    parser.add_argument("--split", type=float, default=0.8, help="train/test split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, help="random seed for dataset sampling")
    parser.add_argument("--table-name", type=str, help="name of the database table for query plan generation")
    parser.add_argument("--embedder", type=str, help='embedding model name (e.g., "bert-base-uncased")')
    parser.add_argument("--top-k", type=int, help="number of nearest neighbors to return from the database")
    parser.add_argument(
        "--rerankers",
        nargs="+",
        type=str,
        help="List of reranker names to use (e.g., --rerankers deepseek_boolean entailment)",
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

    # Apply custom validation
    if args.build and (not args.source or not args.test_dest or not args.train_dest):
        parser.error("--build requires --num, --source, and --dest arguments")

    if args.query_plan and (not args.target_table or not args.embedder or not args.top_k):
        parser.error("--query-plan requires --table-name, --embeder, and --top-k arguments")

    return args


def build_training_dataset(num_examples, source_path, dest_path, seed=None):
    examples = pd.read_json(source_path, lines=True)
    examples = examples.sample(num_examples, random_state=seed)
    examples.to_json(dest_path, orient="records", lines=True)


def build_train_test_split(source_path, train_save_path, test_save_path, seed=42):
    examples = pd.read_json(source_path, lines=True)
    train = examples.sample(frac=0.8, random_state=seed)
    test = examples.drop(train.index)
    train.to_json(train_save_path, orient="records", lines=True)
    test.to_json(test_save_path, orient="records", lines=True)


def train_test_split_nontrivial(path, split=0.8):
    examples = pd.read_json(path, lines=True)
    train = examples.sample(frac=split, random_state=42)
    test = examples.drop(train.index)

    return train, test


def write_train_test_to_file(train: pd.DataFrame, test: pd.DataFrame, path: str):
    train.to_json(path + "train.jsonl", orient="records", lines=True)
    test.to_json(path + "test.jsonl", orient="records", lines=True)


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
        # device: str,
        dataset_path: str,
        target_table: str,
        target_column: str,
        metric: str,
        embedding_model_name: str,
        normalize: bool,
        query_expansion: str = "identity",
        batch_size: int = 16,
        top_k: int = 100,
        strategy: str = None,
        use_index: bool = True,
        probes: int = 16,
        ef_search: int = 1000,
        reranker_to_use: str = None,
        metrics_config: dict[str, float] = None,
        distance_threshold: float = None,
        output_path: str = "experiments/results/",
        output_search_results: bool = False,
    ):
        # Set up configs
        # self.device = device

        """
        Args:
            dataset_path: path to a jsonl file containing lines that hat at least these keys:
                "sent_no_cit": the sentence with inline citations replaced by "[REF]"
                "sent_idx": the index of the sentence in the original record's "body_sentences" list
                "source_doi": "...",
                "
                "pubdate": int (YYYYMMDD)
        """
        # Dataset and results
        try:
            self.dataset = pd.read_json(dataset_path, lines=True)
        except Exception as e:
            raise ValueError(f"Error reading dataset from path '{dataset_path}': {e}")

        # Initialize database
        self.db = MilvusDB()
        self.top_k = top_k
        self.use_index = use_index
        self.probes = probes
        self.ef_search = ef_search
        self.distance_threshold = distance_threshold

        # Experiment configuration
        self.dataset["expanded_query"] = None  # Placeholder for expanded queries
        self.dataset_path = dataset_path
        self.query_results: list[dict] = []
        self.first_rank: int | None = None  # The rank of the first target doi to appear in the results
        self.last_rank: int | None = None  # Rank at which all target DOIs appear in the results
        self.target_table = target_table
        self.target_column = target_column
        self.metric = metric
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.embedder = Embedder.create(
            model_name=embedding_model_name, device=self.device, normalize=normalize, for_queries=True
        )
        self.query_expansion_name = query_expansion
        self.query_expander = get_expander(query_expansion, path_to_data=EXPANSION_DATA_PATH)
        self.reranker_to_use = reranker_to_use
        # self.reranker = get_reranker(reranker_name=reranker_to_use, db=self.db) if reranker_to_use is not None else None
        self.metrics_config = metrics_config
        self.output_path = output_path

        # Strategy for using multiple document / query expansions
        supported_strategies = {"50-50": self.__fifty_fifty_search, "basic": self.__basic_search}
        if strategy in supported_strategies:
            self.strategy = strategy
        else:
            raise ValueError(
                f"'{strategy}' is not a supported strategy. Supported strategies: {[k for k in supported_strategies.keys()]}"
            )
        self.search = supported_strategies[strategy]  # Search function to use in .run()

        # Prepare attributes for results
        self.recall_matrix = None
        self.hitrate_matrix = None
        self.iou_matrix = None
        self.avg_hitrate_at_k = None
        self.avg_iou_at_k = None
        self.avg_recall_at_k = None
        self.best_k_for_iou = None

        # self.stats_by_topk = {k: {"hitrates": [], "jaccards": []} for k in range(1, top_k + 1)}
        # self.jaccard_scores = {threshold: [] for threshold in DISTANCE_THRESHOLDS}
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

        self.output_search_results = output_search_results

    def __postprocess_milvus_results(self, results: list[dict]):
        return [hit.entity | {"distance": hit.distance} for hit in results[0]]

    def __basic_search(self, collection, embedding, example) -> pd.DataFrame:
        example_pubdate = example.get("pubdate").strftime("%Y%m%d")
        results = collection.search(
            data=[embedding],  # TODO is this more efficient if we pass a batch?
            anns_field=self.target_column,
            param={"metric_type": "L2"},  # TODO: parameterize this?
            limit=self.top_k,
            output_fields=["text", "pubdate", "doi"],
            filter=f"pubdate < {example_pubdate}",
        )
        return self.__postprocess_milvus_results(results)

    def __fifty_fifty_search(self, db, embedding, example) -> pd.DataFrame:
        """
        This function retrieves half the top-k from chunks, and half from contributions,
        then returns the interleaved results
        """
        half_k = self.top_k // 2

        chunk_results = db.vector_search(
            query_vector=embedding,
            target_table="chunks",
            target_column="embedding",
            metric=self.metric,
            pubdate=example.get("pubdate"),
            use_index=self.use_index,
            top_k=half_k,
            probes=self.probes,
            ef_search=self.ef_search,
        )

        contribution_results = db.vector_search(
            query_vector=embedding,
            target_table="contributions",
            target_column="embedding",
            metric=self.metric,
            pubdate=example.get("pubdate"),
            use_index=self.use_index,
            top_k=half_k,
            probes=self.probes,
            ef_search=self.ef_search,
        )

        # Combine and sort
        interleaved = []
        for i in range(half_k):
            interleaved.append(chunk_results.iloc[i])
            interleaved.append(contribution_results.iloc[i])

        return pd.DataFrame(interleaved).reset_index(drop=True)

    def __hit_rate(self, example, results: list[dict]):
        target_dois = set(example["citation_dois"])
        retrieved_dois = {result["doi"] for result in results}

        num_hits = len(target_dois.intersection(retrieved_dois))
        return num_hits / len(target_dois)

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

    def __jaccard(self, example: dict, results: list[dict]):
        """
        Takes an 'example' (a dict representing a row from the input dataset) and a list of
        'results' (dicts representing VectorSearchResult objects) and computes the Jaccard score
        between the predicted DOIs and the ground truth DOIs.
        """
        predicted_dois = {result["doi"] for result in results}
        citation_dois = set(example["citation_dois"])
        return self.__jaccard_score(predicted_dois, citation_dois)

    def __compute_stats(self):
        for example, results in tqdm(
            self.query_results, desc="Computing Stats", position=2, total=len(self.query_results)
        ):
            # Compute stats by top k
            for i in range(self.top_k):
                k_idx = i + 1

                # Get the hitrate percent for this example at this k
                hitrate = self.__hit_rate(example, results[:k_idx])
                self.stats_by_topk[k_idx]["hitrates"].append(hitrate)

                # Get the Jaccard score for this example at this k
                jaccard_score = self.__jaccard(example, results[:k_idx])
                self.stats_by_topk[k_idx]["jaccards"].append(jaccard_score)

                # Compute first and last ranks
                example["first_rank"] = None
                example["last_rank"] = None

                target_dois = set(example["citation_dois"])
                # Skip examples with no target DOIs
                if not target_dois:
                    continue

                # Find the first rank
                for i, result in enumerate(results):
                    if result["doi"] in target_dois:
                        example["first_rank"] = i + 1  # +1 because ranks are 1-indexed
                        break

                # If no first rank found, there won't be a last rank either
                if example["first_rank"] is None:
                    continue

                # Special case: only 1 target DOI then first rank is also last rank
                if len(set(example["citation_dois"])) == 1:
                    example["last_rank"] = example["first_rank"]
                    continue

                # Find the last rank: first check the full results if all target DOIs are present
                all_retrieved_dois = {result["doi"] for result in results}
                if not target_dois.issubset(all_retrieved_dois):
                    continue

                retrieved_dois = set()
                for i, result in enumerate(results):
                    retrieved_dois.add(result["doi"])
                    if target_dois.issubset(retrieved_dois):
                        example["last_rank"] = i + 1
                        break

        # Compute summary stats by top k
        for k in self.stats_by_topk:
            avg_jaccard = sum(self.stats_by_topk[k]["jaccards"]) / len(self.stats_by_topk[k]["jaccards"])
            avg_hitrate = sum(self.stats_by_topk[k]["hitrates"]) / len(self.stats_by_topk[k]["hitrates"])
            self.stats_by_topk[k]["avg_jaccard"] = avg_jaccard
            self.stats_by_topk[k]["avg_hitrate"] = avg_hitrate

        # Compute other summary stats?

    def _get_output_filename_base(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        reranker_str = f"_{self.reranker_to_use}" if self.reranker_to_use else ""
        return f"{self.strategy}_{self.target_table}_{self.query_expansion_name}{reranker_str}_norm{self.normalize}_{self.metric_to_str[self.metric]}_n{len(self.dataset)}_{current_time}"

    def __write_run_results(self):
        """
        Writes out the results of a .run() experiment, which only includes the config and the average Jaccard score.
        """
        filename_base = self._get_output_filename_base()

        # Create directory if it doesn't exist
        if not os.path.exists(f"{self.output_path}/{filename_base}"):
            os.makedirs(f"{self.output_path}/{filename_base}")

        output = {
            "config": self.get_config_dict(),
            "average_score": self.average_score,
            "ef_search": self.ef_search,
            "average_hitrate_at_k": self.avg_hitrate_at_k,
            "average_iou_at_k": self.avg_iou_at_k,
            "average_recall_at_k": self.avg_recall_at_k,
            "best_recall_k": max(self.avg_recall_at_k),
            "best_iou_k": max(self.avg_iou_at_k),
        }

        file_path = os.path.join(self.output_path, filename_base, f"results_{filename_base}.json")
        print(f"Writing output to {self.output_path}/{filename_base}")
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

    def get_config_dict(self):
        return {
            "dataset": self.dataset_path,
            "table": self.target_table,
            "target_column": self.target_column,
            "metric": self.metric,
            "embedder": self.embedder.model_name,
            "normalize": self.normalize,
            "query_expansion": self.query_expansion_name,
            "strategy": self.strategy,
            "use_index": self.use_index,
            "rerankers": self.reranker_to_use,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "probes": self.probes,
            "ef_search": self.ef_search,
        }

    # --- New helper methods for serializing results to JSON-friendly primitives ---
    def _normalize_value(self, val):
        """Convert numpy / pandas / datetime types to plain Python types for JSON."""
        try:
            import numpy as _np
            import pandas as _pd
        except Exception:
            _np = None
            _pd = None

        # numpy scalars
        if _np is not None and isinstance(val, (_np.integer, _np.floating, _np.bool_)):
            return val.item()
        # numpy arrays
        if _np is not None and isinstance(val, _np.ndarray):
            return val.tolist()
        # pandas timestamp
        if _pd is not None and isinstance(val, _pd.Timestamp):
            return val.isoformat()
        # datetime
        from datetime import datetime as _dt

        if isinstance(val, _dt):
            return val.isoformat()
        # fallback
        try:
            json_encodable = val if isinstance(val, (str, int, float, bool, type(None))) else str(val)
            return json_encodable
        except Exception:
            return str(val)

    def _serialize_hit(self, hit):
        """
        Take a single 'hit' returned from Milvus/DB and convert to a plain dict with serializable values.
        Handles dict-like hits or objects with common attributes (entity, doi, distance, text, pubdate).
        """
        # If it's already a dict-like mapping, normalize values
        if isinstance(hit, dict):
            return {k: self._normalize_value(v) for k, v in hit.items()}

        # Try typical attributes used in this codebase
        serialized = {}
        try:
            # Some milvus clients expose the entity as hit.entity (dict-like)
            entity = getattr(hit, "entity", None)
            if isinstance(entity, dict):
                for k, v in entity.items():
                    serialized[k] = self._normalize_value(v)
            # Common direct attributes
            for attr in ("doi", "text", "pubdate", "id"):
                if hasattr(hit, attr):
                    serialized[attr] = self._normalize_value(getattr(hit, attr))
            # distance usually present
            if hasattr(hit, "distance"):
                serialized["distance"] = self._normalize_value(getattr(hit, "distance"))
        except Exception:
            # As a last resort, include repr
            serialized["repr"] = repr(hit)

        # If no useful fields found, return repr
        if not serialized:
            return {"repr": repr(hit)}
        return serialized

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

        collection = Collection(name=self.target_table)
        collection.load()
        print(f"Collection {self.target_table} loaded.")

        # Set up rank fusion
        # rank_fuser = RankFuser(config=self.metrics_config) if self.metrics_config else None

        # Create thread-safe queues for tasks and results
        task_queue = queue.Queue()
        results_queue = queue.Queue()
        progress_lock = threading.Lock()

        # For tracking progress
        dataset_size = len(self.dataset)
        consumer_progress = 0
        consumer_bar = None  # Declare consumer bar reference; initialized during producer startup
        sentinel = object()  # Unique sentinel object for signaling completion

        def consumer_thread():
            """Consumer thread that handles database queries and evaluations"""
            nonlocal consumer_progress

            while True:
                item = task_queue.get()
                try:
                    if item is sentinel:
                        break
                    batch_records, embeddings = item

                    # Query the database
                    thread_client = MilvusDB()
                    search_results = thread_client.search(
                        collection_name=self.target_table,
                        query_records=batch_records,
                        query_vectors=embeddings,
                        metric=self.metric,
                        limit=self.top_k,
                    )

                    # TODO: fix logging within thread

                    # Log any anomalies in record retrieval
                    if len(search_results) != len(embeddings):
                        logger.warning(f"Expected {len(embeddings)} results, but got {len(search_results)} for batch")
                        print(f"Expected {len(embeddings)} results, but got {len(search_results)} for batch")
                    if len(search_results[0]) != self.top_k:
                        logger.warning(f"Expected {self.top_k} results, but got {len(search_results[0])} for batch.")
                        print(f"Expected {self.top_k} results, but got {len(search_results[0])} for batch.")

                    # Put the (records, results) pair on queue for stats computation later
                    results_queue.put((batch_records, search_results))

                    # Thread-safe update of progress counter
                    # if consumer_bar is not None:
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
                    batch = self.dataset.iloc[slice(i, i + self.batch_size)]
                    expanded_queries = self.query_expander(batch)
                    embeddings = self.embedder(expanded_queries)

                    # Convert to dicts and put on consumer task_queue
                    batch_records = batch.to_dict(orient="records")
                    task_queue.put((batch_records, embeddings.tolist()))
                    producer_bar.update(len(batch_records))

                # Put sentinels on the task queue to signal consumer completion
                for _ in range(num_workers):
                    task_queue.put(sentinel)

                # Cleanup the producer
                task_queue.join()

        print(f"Experiment computed in {time() - start:.2f} seconds")

        # Initialize the results matrices and compute stats
        start = time()
        self.hitrate_matrix = np.zeros((len(self.dataset), self.top_k))
        self.iou_matrix = np.zeros((len(self.dataset), self.top_k))
        self.recall_matrix = np.zeros((len(self.dataset), self.top_k))
        stats_idx = 0

        # If streaming search results to disk, prepare the output file now (single-threaded write)
        out_file = None
        if self.output_search_results:
            try:
                # Ensure output directory exists
                os.makedirs(self.output_path, exist_ok=True)
                out_path = os.path.join(self.output_path, "search_results.jsonl")
                out_file = open(out_path, "w", encoding="utf-8")
                print(f"Streaming search results to {out_path}")
            except Exception as e:
                logger.error(f"Could not open search results file for writing: {e}")
                out_file = None

        # Drain queue and compute batch stats, writing each record+results row to disk as a JSON line
        while not results_queue.empty():
            batch_records, batch_results = results_queue.get()
            stats = self._compute_metrics_batch(batch_records, batch_results)

            # Insert stats' B * self.top_k arrays onto the corresponding matrices
            self.recall_matrix[stats_idx : stats_idx + len(batch_records), :] = stats["recall_at_k"]
            self.iou_matrix[stats_idx : stats_idx + len(batch_records), :] = stats["iou_at_k"]
            self.hitrate_matrix[stats_idx : stats_idx + len(batch_records), :] = stats["hitrate_at_k"]
            stats_idx += len(batch_records)

            # Stream results line-by-line to file to avoid building a huge in-memory list / DataFrame
            if self.output_search_results and out_file is not None:
                for rec, res in zip(batch_records, batch_results):
                    # try:
                    #     # Normalize the record (record likely already a dict)
                    #     serial_rec = {k: self._normalize_value(v) for k, v in rec.items()}
                    # except Exception:
                    #     serial_rec = {"repr": str(rec)}

                    # Normalize hits
                    # serial_res = []
                    # try:
                    #     for hit in res:
                    #         serial_res.append(self._serialize_hit(hit))
                    # except Exception:
                    #     # If res cannot be iterated (unexpected), store its repr
                    #     serial_res = [{"repr": str(res)}]

                    # Write single json line
                    try:
                        # out_file.write(
                        #     json.dumps({"record": serial_rec, "results": serial_res}, ensure_ascii=False) + "\n"
                        # )
                        out_file.write(json.dumps({"record": rec, "results": res}, ensure_ascii=False) + "\n")
                    except Exception as e:
                        # If writing a particular line fails, log and continue
                        logger.error(f"Failed writing a search-result line: {e}")

        # Close the output file if used
        if out_file is not None:
            try:
                out_file.close()
            except Exception:
                pass

        # Ensure we processed everything we enqueued
        if stats_idx != len(self.dataset):
            logger.warning(f"Stats rows filled ({stats_idx}) != dataset size ({len(self.dataset)}).")

        # Compute summary stats
        self.avg_recall_at_k = self.recall_matrix.mean(axis=0).tolist()
        self.avg_hitrate_at_k = self.hitrate_matrix.mean(axis=0).tolist()
        self.avg_iou_at_k = self.iou_matrix.mean(axis=0).tolist()
        self.best_k_for_iou = int(np.argmax(self.avg_iou_at_k)) + 1  # +1 for 1-indexed k
        print(f"Stats computed in {time() - start:.2f} seconds")
        self.__write_run_results()

    def __str__(self):
        return (
            f"{'='*40}\n"
            f"{'Device':<20}: {self.device}\n"
            f"{'Dataset Path':<20}: {self.dataset_path}\n"
            f"{'Dataset Size':<20}: {len(self.dataset)}\n"
            f"{'Table':<20}: {self.target_table}\n"
            f"{'Target Column':<20}: {self.target_column}\n"
            f"{'Metric':<20}: {self.metric_to_str.get(self.metric, self.metric)}\n"
            f"{'Embedder':<20}: {self.embedder.model_name}\n"
            f"{'Normalize':<20}: {self.normalize}\n"
            f"{'Enrichment':<20}: {self.query_expansion_name}\n"
            f"{'Rank Fusers':<20}: {self.metrics_config}\n"
            f"{'Search Strategy':<20}: {self.strategy}\n"
            f"{'Batch Size':<20}: {self.batch_size}\n"
            f"{'Top k':<20}: {self.top_k}\n"
            f"{'Probes':<20}: {self.probes}\n"
            f"{'ef_search':<20}: {self.ef_search}\n"
            f"{'='*40}\n"
        )

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
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    # Set up logging
    logging.basicConfig(
        filename="logs/experiment.log",
        filemode="w",
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.build:
        source, train_dest, test_dest, split, seed = (
            args.source,
            args.train_dest,
            args.test_dest,
            args.split,
            args.seed,
        )
        print(f"Building dataset from {source}. Split: {split}:{train_dest}, {1 - split}:{test_dest}. Seed: {seed}")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source dataset {source} does not exist.")
        build_train_test_split(
            source_path=source,
            train_save_path=train_dest,
            test_save_path=test_dest,
            seed=seed,
        )
        print(f"Train/test split written to {train_dest} and {test_dest}.")
        return

    if args.run:
        # Load experiment configs
        with open(args.run, "r") as config_file:
            config = yaml.safe_load(config_file)

        experiment = Experiment(
            dataset_path=config["dataset"],
            target_table=config["table"],
            target_column=config["target_column"],
            metric=config.get("metric", "vector_cosine_ops"),
            embedding_model_name=config["embedder"],
            normalize=config["normalize"],
            query_expansion=config["query_expansion"],
            batch_size=config.get("batch_size", 16),
            top_k=config.get("top_k", 100),
            probes=config.get("probes", 16),
            ef_search=config.get("ef_search", 40),
            use_index=config.get("use_index", False),
            strategy=config.get("strategy", "basic"),
            reranker_to_use=config.get("reranker", None),
            metrics_config=config.get("metrics", None),
            # distance_threshold=config["distance_threshold"],
            output_path=config.get("output_path", "experiments/results/"),
            output_search_results=config.get("output_search_results", False),
        )
        print(experiment)
        experiment.run()

        return

    if args.write:
        train, test = train_test_split_nontrivial("data/dataset/full/nontrivial.jsonl")
        write_train_test_to_file(train, test, "data/dataset/split/")
        return


if __name__ == "__main__":
    main()
