import argparse
import json
import logging
import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict
from time import time
from matplotlib.ticker import MultipleLocator
from datetime import datetime
from tqdm import tqdm

from database.database import Database, VectorSearchResult
from query_expander import get_expander
from Embedders import get_embedder

logger = logging.getLogger(__name__)
DISTANCE_THRESHOLDS = np.arange(1.0, 0.0, -0.01)

EXPANSION_DATA_PATH = "data/preprocessed/reviews.jsonl"


def argument_parser():
    """
    Example usage:

    1. Run an experiment with specified configuration:
       python run_experiment.py --run experiments/configs/bert_cosine.yaml

    2. Build a train/test split by sampling from source:
       python run_experiment.py --build --source data/dataset/full/nontrivial_llm.jsonl --train-dest data/dataset/sampled/train.jsonl --test-dest data/dataset/sample/test.jsonl --split=0.8 --seed 42

    3. Run an experiment with a top-k scan:
       python run_experiment.py --run-scan experiments/bert_cosine.yaml

    4. Generate query plans and analyze database performance:
       python run_experiment.py --query-plan --table-name bert_hnsw --embedder bert-base-uncased --top-k 50

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

    # @classmethod
    # def create_all_experiments(model_name, dataset, top_k, probes):

    metric_to_str = {"vector_l2_ops": "L2", "vector_cosine_ops": "cosine", "vector_ip_ops": "ip"}

    def __init__(
        self,
        device: str,
        dataset_path: str,
        target_table: str,
        target_column: str,
        metric: str,
        embedding_model_name: str,
        normalize: bool,
        query_expansion: str = "identity",
        batch_size: int = 16,
        top_k: int = 100,
        use_index: bool = True,
        probes: int = 16,
        ef_search: int = 1000,
        rerankers_to_use=[],
        distance_threshold: float = None,
    ):
        # Set up configs
        self.device = device

        """
        NOTE: nontrivial_train.index.tolist() will give the line numbers of the original example
        so we can look up the original sentence, etc.
        """
        # Dataset and results
        self.dataset = pd.read_json(dataset_path, lines=True)
        self.dataset_path = dataset_path
        self.query_results: list[dict] = []

        self.target_table = target_table
        self.target_column = target_column
        self.metric = metric
        self.batch_size = batch_size
        self.normalize = normalize
        self.embedder = get_embedder(model_name=embedding_model_name, device=device, normalize=normalize)
        self.query_expansion_name = query_expansion
        self.query_expander = get_expander(query_expansion, path_to_data=EXPANSION_DATA_PATH)

        # Initialize database
        self.db = Database(path_to_env=".env")
        self.db.test_connection()
        self.top_k = top_k
        self.use_index = use_index
        self.probes = probes
        self.ef_search = ef_search
        self.rerankers_to_use = rerankers_to_use
        self.distance_threshold = distance_threshold

        # Prepare attributes for results
        self.stats_by_topk = {k: {"hitrates": [], "jaccards": []} for k in range(1, top_k + 1)}
        self.jaccard_scores = {threshold: [] for threshold in DISTANCE_THRESHOLDS}
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

    def __hit_rate(self, example, results: pd.DataFrame):
        target_dois = set(example["citation_dois"])
        retrieved_dois = set(results["doi"])

        num_hits = len(target_dois.intersection(retrieved_dois))
        return num_hits / len(target_dois)
        # return {
        #     "pct": num_hits / len(target_dois),
        #     "num_hits": num_hits,
        #     "total_targets": len(target_dois),
        # }

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

    def __jaccard(self, example: pd.Series, results: pd.DataFrame):
        """
        Takes an 'example' (a pd.Series representing a row from the input dataset) and a list of
        'results' (VectorSearchResult objects) and computes the Jaccard score between the predicted DOIs
        and the ground truth DOIs.
        """
        predicted_dois = set(results["doi"])
        citation_dois = set(example["citation_dois"])
        score = self.__jaccard_score(predicted_dois, citation_dois)
        return score

    def __plot_topk_histogram(self, filename_base: str):
        outfile = f"experiments/results/{filename_base}/topk_histogram_{filename_base}.png"

        plt.figure()
        plt.hist(
            self.best_top_ks,
            #  bins=30,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        plt.xlabel("Best Top-k")
        plt.ylabel("Frequency")
        plt.title(f"Best Top-k value (n = {len(self.dataset)})")
        plt.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.7)
        plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

        plt.savefig(outfile)
        plt.close()

    def __plot_roc_curve(self, filename_base: str):
        outfile = f"experiments/results/{filename_base}/roc_{filename_base}.png"

        plt.figure()
        thresholds = sorted(self.average_scores.keys())
        avg_scores = [self.average_scores[threshold] for threshold in thresholds]
        plt.plot(thresholds, avg_scores, marker=".", linestyle="-", label="Average Jaccard Score")
        plt.xlabel(f"Distance Threshold (n = {len(self.dataset)})")

        # Calculate average values
        avg_best_distance = (
            sum(self.best_top_distances) / len(self.best_top_distances) if self.best_top_distances else 0
        )
        avg_best_top_k = sum(self.best_top_ks) / len(self.best_top_ks) if self.best_top_ks else 0

        # Add a text box with metrics in the top-right corner
        textstr = f"Avg Best Distance: {avg_best_distance:.4f}\nAvg Best Top-k: {int(avg_best_top_k)}"
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")

        # Place the text box in the top right (coordinates are in axes space: 0,0 is bottom left, 1,1 is top right)
        plt.text(
            0.95,
            0.95,
            textstr,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        plt.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.7)
        plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
        plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))

        plt.savefig(outfile)
        plt.close()

    def _get_output_filename_base(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.target_table}_{self.query_expansion_name}_norm{self.normalize}_{self.metric_to_str[self.metric]}_n{len(self.dataset)}_{current_time}"

    # def __write_json_results(self, filename_base):
    #     # Prep results and outfile name
    #     output = {
    #         "config": self.get_config_dict(),
    #         "averages": self.average_scores,
    #         "avg_best_top_k": sum(self.best_top_ks) / len(self.best_top_ks),
    #         "avg_best_distance_threshold": sum(self.best_top_distances) / len(self.best_top_distances),
    #         "best_top_ks": self.best_top_ks,
    #     }

    #     # Create directory if it doesn't exist
    #     if not os.path.exists(f"experiments/results/{filename_base}"):
    #         os.makedirs(f"experiments/results/{filename_base}")

    #     # Write and plot results
    #     with open(f"experiments/results/{filename_base}/results_{filename_base}.json", "w") as f:
    #         json.dump(output, f)

    def __write_run_results(self):
        """
        Writes out the results of a .run() experiment, which only includes the config and the average Jaccard score.
        """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{self.target_table}_{self.query_expansion_name}_norm{self.normalize}_n{len(self.dataset)}_topk{self.top_k}_{current_time}"

        # Create directory if it doesn't exist
        if not os.path.exists(f"experiments/results/{filename_base}"):
            os.makedirs(f"experiments/results/{filename_base}")

        query_results_json = []
        for example, results in self.query_results:
            results_dict = results.to_dict(orient="records")
            for result in results_dict:
                result["pubdate"] = result["pubdate"].strftime("%Y-%m-%d")
            query_results_json.append([example, results_dict])
        with open(f"experiments/results/{filename_base}/query_results_{filename_base}.json", "w") as f:
            json.dump(query_results_json, f)

        # avg_hit_pct = sum(hit["pct"] for hit in self.hits) / len(self.hits) if self.hits else 0
        # Note that k here are keys from 1 to top_k, so the list index = k - 1
        avg_jaccards = [
            sum(self.stats_by_topk[k]["jaccards"]) / len(self.stats_by_topk[k]["jaccards"]) for k in self.stats_by_topk
        ]
        average_hit_rates = [self.stats_by_topk[k]["avg_hitrate"] for k in self.stats_by_topk]

        output = {
            "config": self.get_config_dict(),
            "average_score": self.average_score,
            "ef_search": self.ef_search,
            "best_top_ks": self.best_top_ks,
            "average_hit_rates": average_hit_rates,
            "average_jaccards": avg_jaccards,
        }

        with open(f"experiments/results/{filename_base}/results_{filename_base}.json", "w") as f:
            json.dump(output, f)

        self.__plot_results(filename_base, output)

    def __plot_results(self, filename_base, output):
        import matplotlib.pyplot as plt

        k_values = [k for k in range(1, self.top_k + 1)]

        # Make a plot of the average hit rates (y-axis) and IoU (Jaccard) vs. top-k (x-axis)
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, output["average_hit_rates"], marker="o", label="Average Hit Rate")
        plt.plot(k_values, output["average_jaccards"], marker="o", label="Average Jaccard")
        plt.xlabel("Top-k")
        plt.ylabel("Score")
        plt.title("Average Hit Rates and IoU vs. Top-k")
        plt.legend()
        plt.grid()
        plt.savefig(f"experiments/results/{filename_base}/hitrate_vs_k_{filename_base}.png")
        plt.close()

        # Make a plot of the average Jaccard scores (y-axis) vs. top-k (x-axis)
        # plt.figure(figsize=(10, 6))
        # plt.plot(k_values, output["average_jaccards"], marker="o", label="Average Jaccard")
        # plt.xlabel("Top-k")
        # plt.ylabel("Score")
        # plt.title("Average Jaccard Scores vs. Top-k")
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"experiments/results/{filename_base}/jaccard_vs_k_{filename_base}.png")
        # plt.close()

    # def __write_results(self):
    #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename_base = f"{self.target_column}_{self.query_expansion_name}_norm{self.normalize}_n{len(self.dataset)}_topk{self.top_k}_{current_time}"

    #     # Create directory if it doesn't exist
    #     if not os.path.exists(f"experiments/results/{filename_base}"):
    #         os.makedirs(f"experiments/results/{filename_base}")

    #     # Write and plot results
    #     self.__write_json_results(filename_base)
    #     self.__plot_roc_curve(filename_base)
    #     self.__plot_topk_histogram(filename_base)

    def get_config_dict(self):
        return {
            "dataset": self.dataset_path,
            "table": self.target_table,
            "target_column": self.target_column,
            "metric": self.metric,
            "embedder": self.embedder.model_name,
            "normalize": self.normalize,
            "query_expansion": self.query_expansion_name,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "probes": self.probes,
            "ef_search": self.ef_search,
        }

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

        # Set up database for efficient queries
        self.db.set_session_resources(optimize_for="query", verbose=False)
        self.db.prewarm_table(self.target_table, target_column=self.target_column)

        # Create thread-safe queues for tasks and results
        task_queue = queue.Queue(maxsize=20)
        results_queue = queue.Queue()
        progress_lock = threading.Lock()
        query_bar_lock = threading.Lock()

        # For tracking progress
        dataset_size = len(self.dataset)
        producer_progress = 0
        consumer_progress = 0

        def consumer_thread():
            """Consumer thread that handles database queries and evaluations"""
            nonlocal consumer_progress

            # Create a dedicated database connection for this thread
            thread_db = Database(path_to_env=".env")
            thread_db.set_session_resources(optimize_for="query", verbose=False)

            while True:
                try:
                    # Get task from queue
                    task = task_queue.get(timeout=60)
                    if task is None:  # Sentinel to signal completion
                        task_queue.task_done()
                        break

                    example, embedding = task

                    # Query the database
                    results = thread_db.vector_search(
                        query_vector=embedding,
                        target_table=self.target_table,
                        target_column=self.target_column,
                        metric=self.metric,
                        pubdate=example.get("pubdate"),
                        use_index=self.use_index,
                        top_k=self.top_k,
                        probes=self.probes,
                        ef_search=self.ef_search,
                    )

                    # TODO: Reranking logic will go here

                    if len(results) != self.top_k:
                        logger.warning(f"Expected {self.top_k} results, but got {len(results)}. ")
                    results_queue.put((example, results))

                    # Compute hit rate and IoU for each k from 1 to top_k
                    # TODO: Move all stats computation to a separate method and put it at the end

                    # Thread-safe update of progress counter
                    with progress_lock, query_bar_lock:
                        consumer_progress += 1
                        query_bar.update(1)  # Update progress bar directly

                    task_queue.task_done()

                except queue.Empty:
                    print("Consumer timeout waiting for tasks")
                    break
                except Exception as e:
                    print(f"Consumer thread error: {str(e)}")
                    task_queue.task_done()

        # Start consumer threads
        num_workers = max(1, num_cpus - 1)  # Leave one core for the main thread
        print(f"Starting {num_workers} database query workers")

        start = time()
        consumer_threads = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Start consumer threads
            for _ in range(num_workers):
                thread = executor.submit(consumer_thread)
                consumer_threads.append(thread)

            # Create progress bars
            with tqdm(total=dataset_size, desc="Embedding (GPU)", position=0) as embed_bar, tqdm(
                total=dataset_size, desc="DB Queries", position=1
            ) as query_bar:

                # Main thread acts as the producer
                for i in range(0, dataset_size, self.batch_size):
                    # GPU cache clearing
                    if i % 50 == 0 and self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif i % 50 == 0 and self.device == "mps":
                        torch.mps.empty_cache()

                    # Get batch, perform any query expansion & generate embeddings
                    batch = self.dataset.iloc[i : i + self.batch_size]
                    expanded_queries = self.query_expander(batch)
                    embeddings = self.embedder(expanded_queries)

                    # Add tasks to queue and update producer progress
                    for j in range(len(batch)):
                        example_dict = batch.iloc[j].to_dict()
                        embedding = embeddings[j]
                        task_queue.put((example_dict, embedding))

                        # Update producer progress
                        producer_progress += 1
                        embed_bar.update(1)

                    # Check and update consumer progress
                    current_consumer = consumer_progress  # Read once to avoid race conditions
                    if current_consumer > query_bar.n:
                        query_bar.update(current_consumer - query_bar.n)

                # Add sentinel values to signal consumer completion
                for _ in range(num_workers):
                    task_queue.put(None)

                # Wait for all tasks to be processed
                task_queue.join()

                # Final consumer progress update
                if consumer_progress > query_bar.n:
                    query_bar.update(consumer_progress - query_bar.n)

        # Append the (example dict, results list[dict]) pairs
        while not results_queue.empty():
            self.query_results.append(results_queue.get())

        print(f"Experiment computed in {time() - start:.2f} seconds")
        start = time()

        # Compute stats
        for example, results in tqdm(
            self.query_results, desc="Computing Stats", position=2, total=len(self.query_results)
        ):
            # Compute stats for each example
            # self.num_results.append(len(results))
            # self.best_top_ks.append(min(self.top_k, len(results)))
            # self.best_top_distances.append(results[self.best_top_ks[-1] - 1].distance if results else float("inf"))
            # self.hits.append(self.__hit_rate(example, results))

            # # Compute Jaccard score for this example
            # jaccard_score = self.__jaccard(example, results)
            # self.jaccard_scores[jaccard_score].append(jaccard_score)

            for i in range(self.top_k):
                k_idx = i + 1

                # Get the hitrate percent for this example at this k
                hitrate = self.__hit_rate(example, results[:k_idx])
                self.stats_by_topk[k_idx]["hitrates"].append(hitrate)

                # Get the Jaccard score for this example at this k
                jaccard_score = self.__jaccard(example, results[:k_idx])
                self.stats_by_topk[k_idx]["jaccards"].append(jaccard_score)

        # Compute stats
        for k in self.stats_by_topk:
            avg_jaccard = sum(self.stats_by_topk[k]["jaccards"]) / len(self.stats_by_topk[k]["jaccards"])
            avg_hitrate = sum(self.stats_by_topk[k]["hitrates"]) / len(self.stats_by_topk[k]["hitrates"])
            self.stats_by_topk[k]["avg_jaccard"] = avg_jaccard
            self.stats_by_topk[k]["avg_hitrate"] = avg_hitrate

        print(f"Stats computed in {time() - start:.2f} seconds")

        self.__write_run_results()

    def __str__(self):
        return (
            f"Experiment Configuration:\n"
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
            f"{'Batch Size':<20}: {self.batch_size}\n"
            f"{'Top k':<20}: {self.top_k}\n"
            f"{'Probes':<20}: {self.probes}\n"
            f"{'ef_search':<20}: {self.ef_search}\n"
            f"{'='*40}\n"
        )


def main():

    args = argument_parser()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

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

        # Set up and run experiment
        experiment = Experiment(
            device=device,
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
            # rerankers=config.get("rerankers", []),
            # distance_threshold=config["distance_threshold"],
        )
        print(experiment)
        experiment.run()

        return

    if args.run_scan:
        # Load expermient configs
        with open(args.run_scan, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Set up and run experiment
        experiment = Experiment(
            device=device,
            dataset_path=config["dataset"],
            target_table=config.get("table", "lib"),
            target_column=config.get("target_column", "chunk"),
            metric=config.get("metric", "vector_cosine_ops"),
            embedding_model_name=config["embedder"],
            normalize=config["normalize"],
            query_expansion=config["enrichment"],
            batch_size=config.get("batch_size", 16),
            top_k=config.get("top_k"),
            probes=config.get("probes", 40),
            ef_search=config.get("ef_search", 40),
        )
        print(experiment)
        experiment.run_and_topk_scan()

        return

    if args.write:
        train, test = train_test_split_nontrivial("data/dataset/full/nontrivial.jsonl")
        write_train_test_to_file(train, test, "data/dataset/split/")
        return

    if args.query_plan:
        # Set up resources
        embedder = get_embedder(args.embedder, device=device)
        db = Database()
        db.test_connection()

        # Generate query vector and query plan
        embedding = embedder(["dummy text"])
        embedding = embedding[0]
        print(f"Query vector shape: {embedding.shape}")
        print(f"type(embedding): {type(embedding)}")
        # Print configuration table
        print("\n" + "=" * 60)
        print("QUERY PLAN CONFIGURATION")
        print("=" * 60)
        print(f"{'Parameter':<15} {'Value':<40}")
        print("-" * 60)
        print(f"{'Embedder':<15} {args.embedder:<40}")
        print(f"{'Device':<15} {device:<40}")
        print(f"{'Table Name':<15} {args.target_table:<40}")
        print(f"{'Metric':<15} {'vector_cosine_ops':<40}")
        print(f"{'Top K':<15} {args.top_k:<40}")
        print(f"{'Probes':<15} {args.probes:<40}")
        print(f"{'Batch Size':<15} {args.batch_size:<40}")
        print("=" * 60 + "\n")

        db.prewarm_table(args.target_table)
        db.explain_analyze(
            query_vector=embedding,
            target_table=args.target_table,
            metric="vector_cosine_ops",
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
