import argparse
import json
import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from database.database import Database, VectorQueryResult
from TextEnrichers import get_enricher
from Embedders import get_embedder

from time import time

DISTANCE_THRESHOLDS = np.arange(1.0, 0.0, -0.01)


def argument_parser():
    """
    Example usage:

    1. Run an experiment with specified configuration:
       python run_experiment.py --run --config experiments/configs/bert_cosine.yaml

    2. Build a dataset by sampling from source:
       python run_experiment.py --build --num 1000 --source data/dataset/full/nontrivial.jsonl --dest data/dataset/sampled/sample_1000.jsonl --seed 42

    3. Write out train/test split from a dataset:
       python run_experiment.py --write

    4. Generate query plans and analyze database performance:
       python run_experiment.py --query-plan --table-name bert_hnsw --embedder bert-base-uncased --top-k 50

    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run an experiment with specified configuration or build a dataset."
    )

    # Create mutually exclusive operation groups
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument("--run", action="store_true", help="run the experiment")
    operation_group.add_argument(
        "--run-threaded", action="store_true", help="run the experiment with threading"
    )
    operation_group.add_argument("--build", action="store_true", help="build a dataset")
    operation_group.add_argument(
        "--write", action="store_true", help="write out train/test datasets"
    )
    operation_group.add_argument(
        "--query-plan", action="store_true", help="generate EXPLAIN/ANALYZE query plan for database"
    )

    parser.add_argument("--config", type=str, help="Path to the YAML configuration file.")

    # Dataset building arguments
    parser.add_argument("--num", type=int, help="number of examples to include")
    parser.add_argument("--source", type=str, help="path to source dataset (jsonl)")
    parser.add_argument("--dest", type=str, help="path to destination dataset (jsonl)")
    parser.add_argument("--seed", type=int, help="random seed for dataset sampling")
    parser.add_argument(
        "--table-name", type=str, help="name of the database table for query plan generation"
    )
    parser.add_argument(
        "--embedder", type=str, help='embedding model name (e.g., "bert-base-uncased")'
    )
    parser.add_argument(
        "--top-k", type=int, help="number of nearest neighbors to return from the database"
    )

    args = parser.parse_args()

    # Apply custom validation
    if args.run and not args.config:
        parser.error("--run requires --config")

    if args.build and (not args.num or not args.source or not args.dest):
        parser.error("--build requires --num, --source, and --dest arguments")

    if args.query_plan and (not args.table_name or not args.embedder or not args.top_k):
        parser.error("--query-plan requires --table-name, --embeder, and --top-k arguments")

    return args


def build_training_dataset(num_examples, source_path, dest_path, seed=None):
    examples = pd.read_json(source_path, lines=True)
    examples = examples.sample(num_examples, random_state=seed)
    examples.to_json(dest_path, orient="records", lines=True)


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
        table: str,
        target_column: str,
        metric: str,
        embedding_model_name: str,
        normalize: bool,
        enrichment: str,
        batch_size: int = 16,
        top_k: int = 100,
        probes: int = 16,
    ):
        # Set up configs
        self.device = device

        """
        NOTE: nontrivial_train.index.tolist() will give the line numbers of the original example
        so we can look up the original sentence, etc.
        """
        self.dataset = pd.read_json(dataset_path, lines=True)
        self.dataset_path = dataset_path

        self.table = table
        self.target_column = target_column
        self.metric = metric
        self.batch_size = batch_size
        self.embedder = get_embedder(
            model_name=embedding_model_name, device=device, normalize=normalize
        )
        self.normalize = normalize
        self.enrichment = enrichment
        self.enricher = get_enricher(enrichment, path_to_data="data/preprocessed/reviews.jsonl")

        # Initialize database
        self.db = Database(path_to_env=".env")
        self.db.test_connection()
        self.top_k = top_k
        self.probes = probes

        # Prepare attributes for results
        self.jaccard_scores = {threshold: [] for threshold in DISTANCE_THRESHOLDS}
        """
        Dictionary of average Jaccard scores for each distance threshold
        {0.5: 0.1785} means after only keeping query results with distance < 0.5, the average IoU score for
        all examples in the dataset is 0.1785
        """
        self.average_scores = {}
        self.best_top_ks = []
        self.best_top_distances = []

    def _closest_neighbors(self, results, threshold: float):
        """
        Assumes that `results` are ordered by distance, lowest to highest.

        Returns only the results that have distance below the threshold
        """
        for i in range(len(results)):
            if results[i].distance > threshold:
                return results[:i]
        return results

    def __compute_stats(
        self, target_dois: list[str], results: list[VectorQueryResult]
    ) -> list[np.longdouble]:
        """
        NOTE: this assumes that target_dois is non-empty
        """
        target_dois_set = set(target_dois)
        jaccard_scores = []
        for i in range(len(results)):
            predicted_dois = set(result.doi for result in results[: i + 1])
            score = self.__jaccard_similarity(predicted_dois, target_dois_set)
            jaccard_scores.append(score)

    def _evaluate_prediction(self, example, results):
        unique_predicted_dois = set(result.doi for result in results)
        citation_dois = set(doi for doi in example.citation_dois)
        score = self.__jaccard_similarity(unique_predicted_dois, citation_dois)
        return score

    def __jaccard_similarity(self, set1, set2):
        intersection = np.longdouble(len(set1.intersection(set2)))
        union = np.longdouble(len(set1.union(set2)))
        return intersection / union

    def _plot_roc_curve(self, filename_base: str):
        outfile = f"experiments/results/{filename_base}/roc_{filename_base}.png"

        plt.figure()
        thresholds = sorted(self.average_scores.keys())
        avg_scores = [self.average_scores[threshold] for threshold in thresholds]
        plt.plot(thresholds, avg_scores, marker=".", linestyle="-", label="Average Jaccard Score")
        plt.xlabel(f"Distance Threshold (n = {len(self.dataset)})")

        # Calculate average values
        avg_best_distance = (
            sum(self.best_top_distances) / len(self.best_top_distances)
            if self.best_top_distances
            else 0
        )
        avg_best_top_k = sum(self.best_top_ks) / len(self.best_top_ks) if self.best_top_ks else 0

        # Add a text box with metrics in the top-right corner
        textstr = (
            f"Avg Best Distance: {avg_best_distance:.4f}\nAvg Best Top-k: {int(avg_best_top_k)}"
        )
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
        return f"{self.table}_{self.enrichment}_norm{self.normalize}_{self.metric_to_str[self.metric]}_n{len(self.dataset)}_{current_time}"

    def _write_json_results(self, filename_base):
        # Prep results and outfile name
        output = {
            "config": self.get_config_dict(),
            "averages": self.average_scores,
            "avg_best_top_k": sum(self.best_top_ks) / len(self.best_top_ks),
            "avg_best_distance_threshold": sum(self.best_top_distances)
            / len(self.best_top_distances),
        }

        # Create directory if it doesn't exist
        if not os.path.exists(f"experiments/results/{filename_base}"):
            os.makedirs(f"experiments/results/{filename_base}")

        # Write and plot results
        with open(f"experiments/results/{filename_base}/results_{filename_base}.json", "w") as f:
            json.dump(output, f)

    def _write_results(self):
        # Prep results and outfile name
        # metric_str = self.metric_to_str.get(self.metric)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{self.target_column}_{self.enrichment}_norm{self.normalize}_n{len(self.dataset)}_topk{self.top_k}_{current_time}"

        # Create directory if it doesn't exist
        if not os.path.exists(f"experiments/results/{filename_base}"):
            os.makedirs(f"experiments/results/{filename_base}")

        # Write and plot results
        self._write_json_results(filename_base)
        self._plot_roc_curve(filename_base)

    def get_config_dict(self):
        return {
            "dataset": self.dataset_path,
            "table": self.table,
            "target_column": self.target_column,
            "metric": self.metric,
            "embedder": self.embedder.model_name,
            "normalize": self.normalize,
            "enrichment": self.enrichment,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
        }

    def run(self):
        # Load the vector table into memory (shared_buffer) on db host
        self.db.set_session_resources(optimize_for="query")
        self.db.prewarm_table(self.table, target_column=self.target_column)

        for i in tqdm(
            range(0, len(self.dataset), self.batch_size), desc="Batch number", leave=True
        ):
            if i % 50 == 0:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()

            # Enrich and embed batch
            batch = self.dataset.iloc[i : i + self.batch_size]
            enriched_batch = self.enricher(batch)
            # print(f"Iteration {i+1}: Enriching batch of size {len(batch)}")
            embeddings = self.embedder(enriched_batch)

            # TODO: implement batch querying?

            for j in tqdm(range(len(batch)), desc="Querying database", leave=False):
                example = batch.iloc[j]
                this_embedding = embeddings[j]
                results = self.db.query_vector_column(
                    query_vector=this_embedding,
                    table_name=self.table,
                    target_column=self.target_column,
                    metric=self.metric,
                    pubdate=example["pubdate"],
                    use_index=True,
                    top_k=self.top_k,  # 2167399 total chunks
                    probes=self.probes,
                )

                # Compute IoU scores for each distance threshold, find best distance cutoff
                best_score = 0
                best_threshold = 0
                for threshold in DISTANCE_THRESHOLDS:
                    predicted_chunks = self._closest_neighbors(results, threshold)
                    score = self._evaluate_prediction(example, predicted_chunks)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                    self.jaccard_scores[threshold].append(score)

                # Find the top-k corresponding to best threshold
                for i in range(len(results)):
                    if results[i].distance > best_threshold:
                        self.best_top_ks.append(i + 1)  # i is 0-indexed, top k is 1-indexed
                        self.best_top_distances.append(best_threshold)
                        break
                self.best_top_ks.append(i)

        # Compute average scores and write out results
        self.average_scores = {
            round(threshold, 2): float(sum(scores) / len(scores))
            for threshold, scores in self.jaccard_scores.items()
        }
        self._write_results()

    def run_threaded(self):
        assert (
            "CPUS" in os.environ
        ), "CPUS environment variable not set. Please set it to the number of CPU cores available."
        from concurrent.futures import ThreadPoolExecutor
        import queue

        try:
            num_cpus = int(os.getenv("CPUS"))
        except ValueError:
            raise ValueError(
                f"Invalid value for CPUS environment variable (CPUS={num_cpus}). Please set it to an integer."
            )


        self.db.set_session_resources(optimize_for="query")
        self.db.prewarm_table(self.table, target_column=self.target_column)

        # Create queues
        task_queue = queue.Queue()
        results_queue = queue.Queue()
        progress_update_queue = queue.Queue()
        num_completed = 0

        # Consumer function to run database queries and statistics
        def consumer():
            # Create a dedicated database connection for this thread
            thread_db = Database(path_to_env=".env")
            thread_db.set_session_resources(optimize_for="query", verbose=False)

            while True:
                try:
                    # Get task from queue
                    task = task_queue.get()
                    if task is None:  # Sentinel to signal completion
                        task_queue.task_done()
                        break

                    example, embedding = task

                    # Query the database
                    results = thread_db.query_vector_column(
                        query_vector=embedding,
                        table_name=self.table,
                        target_column=self.target_column,
                        metric=self.metric,
                        pubdate=example["pubdate"],
                        use_index=True,
                        top_k=self.top_k,
                        probes=self.probes,
                    )

                    # Compute IoU scores for each distance threshold, find best distance cutoff
                    best_score = 0
                    best_threshold = 0
                    threshold_scores = {}
                    best_top_k = 0
                    best_distance = 0

                    for threshold in DISTANCE_THRESHOLDS:
                        predicted_chunks = self._closest_neighbors(results, threshold)
                        score = self._evaluate_prediction(example, predicted_chunks)
                        threshold_scores[threshold] = score

                        if score > best_score:
                            best_score = score
                            best_threshold = threshold

                    # Find the top-k corresponding to best threshold
                    for i in range(len(results)):
                        if results[i].distance > best_threshold:
                            best_top_k = i + 1  # i is 0-indexed, top k is 1-indexed
                            best_distance = best_threshold
                            break
                    else:  # No break occurred
                        best_top_k = len(results)

                    progress_update_queue.put(True)

                    # Put results in results queue
                    results_queue.put(
                        {
                            "threshold_scores": threshold_scores,
                            "best_top_k": best_top_k,
                            "best_distance": best_distance,
                        }
                    )
                    task_queue.task_done()

                except Exception as e:
                    print(f"Consumer thread error: {e}")
                    progress_update_queue.put(False)
                    task_queue.task_done()

        # Start consumer threads
        num_workers = max(1, num_cpus - 1)  # Adjust based on your system
        print(f"Starting {num_workers} database query workers")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                executor.submit(consumer)

            # 3) Producer section in the main thread, with two progress bars:
            dataset_size = len(self.dataset)
            with tqdm(total=dataset_size, desc="Embedding", position=0) as enqueue_bar, tqdm(
                total=dataset_size, desc="Querying", position=1
            ) as process_bar:

                # Producer enqueues tasks
                for i in range(0, dataset_size, self.batch_size):
                    # GPU cache clearing
                    if i % 50 == 0 and self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif i % 50 == 0 and self.device == "mps":
                        torch.mps.empty_cache()

                    # Enrich + embed
                    batch = self.dataset.iloc[i : i + self.batch_size]
                    enriched_batch = self.enricher(batch)
                    embeddings = self.embedder(enriched_batch)

                    # Enqueue tasks, update "Queueing tasks" bar
                    for j in range(len(batch)):
                        example = batch.iloc[j]
                        embedding = embeddings[j]
                        task_queue.put((example, embedding))
                        enqueue_bar.update(1)

                        while not progress_update_queue.empty():
                            success = progress_update_queue.get()
                            if success:
                                process_bar.update(1)

                        # Process completed results after EACH example is enqueued
                        while not results_queue.empty():
                            result = results_queue.get()
                            # Update stats
                            for threshold, score in result["threshold_scores"].items():
                                self.jaccard_scores[threshold].append(score)
                            self.best_top_ks.append(result["best_top_k"])
                            self.best_top_distances.append(result["best_distance"])
                            num_completed += 1

                # 4) Signal consumers to stop, then wait for them
                for _ in range(num_workers):
                    task_queue.put(None)
                task_queue.join()

                # Process leftover results
                while not results_queue.empty():
                    result = results_queue.get()
                    for threshold, score in result["threshold_scores"].items():
                        self.jaccard_scores[threshold].append(score)
                    self.best_top_ks.append(result["best_top_k"])
                    self.best_top_distances.append(result["best_distance"])
                    num_completed += 1
                    if num_completed <= dataset_size:
                        process_bar.update(1)

        # 5) Final stats + writing results (unchanged)
        self.average_scores = {
            round(threshold, 2): float(sum(scores) / len(scores))
            for threshold, scores in self.jaccard_scores.items()
        }
        self._write_results()

    def __str__(self):
        return (
            f"Experiment Configuration:\n"
            f"{'='*40}\n"
            f"{'Device':<20}: {self.device}\n"
            f"{'Dataset Path':<20}: {self.dataset_path}\n"
            f"{'Dataset Size':<20}: {len(self.dataset)}\n"
            f"{'Table':<20}: {self.table}\n"
            f"{'Target Column':<20}: {self.target_column}\n"
            f"{'Metric':<20}: {self.metric_to_str.get(self.metric, self.metric)}\n"
            f"{'Embedder':<20}: {self.embedder.model_name}\n"
            f"{'Normalize':<20}: {self.normalize}\n"
            f"{'Enrichment':<20}: {self.enrichment}\n"
            f"{'Batch Size':<20}: {self.batch_size}\n"
            f"{'Top k':<20}: {self.top_k}\n"
            f"{'='*40}\n"
        )


def main():
    args = argument_parser()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    if args.build:
        num, source, dest, seed = args.num, args.source, args.dest, args.seed
        print(f"Building dataset with {num} examples from {source} to {dest}. Using seed: {seed}")
        build_training_dataset(num_examples=num, source_path=source, dest_path=dest, seed=seed)
        print("Dataset built.")
        return

    if args.run:
        # Load expermient configs
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Set up and run experiment
        experiment = Experiment(
            device=device,
            dataset_path=config["dataset"],
            table=config.get("table", "lib"),
            target_column=config.get("target_column", "chunk"),
            metric=config.get("metric", "vector_cosine_ops"),
            embedding_model_name=config["embedder"],
            normalize=config["normalize"],
            enrichment=config["enrichment"],
            batch_size=config.get("batch_size", 16),
            top_k=config.get("top_k"),
            probes=config.get("probes"),
        )
        print(experiment)
        experiment.run()
        return

    if args.run_threaded:
        # Load expermient configs
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Set up and run experiment
        experiment = Experiment(
            device=device,
            dataset_path=config["dataset"],
            table=config.get("table", "lib"),
            target_column=config.get("target_column", "chunk"),
            metric=config.get("metric", "vector_cosine_ops"),
            embedding_model_name=config["embedder"],
            normalize=config["normalize"],
            enrichment=config["enrichment"],
            batch_size=config.get("batch_size", 16),
            top_k=config.get("top_k"),
            probes=config.get("probes"),
        )
        print(experiment)
        experiment.run_threaded()
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
        print(f"{'Table Name':<15} {args.table_name:<40}")
        print(f"{'Metric':<15} {'vector_cosine_ops':<40}")
        print(f"{'Top K':<15} {args.top_k:<40}")
        print(f"{'Probes':<15} {args.probes:<40}")
        print(f"{'Batch Size':<15} {args.batch_size:<40}")
        print("=" * 60 + "\n")

        db.prewarm_table(args.table_name)
        db.explain_analyze(
            query_vector=embedding,
            table_name=args.table_name,
            metric="vector_cosine_ops",
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
