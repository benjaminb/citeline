import argparse
import json
import os
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from citeline.statistics import compute_individual_stats, compute_stat_matrices, compute_averages

load_dotenv('../../.env')
assert os.getenv('CPUS'), "Please set CPUS in your .env file"
NUM_WORKERS = int(os.getenv('CPUS')) - 1

def argument_parser():
    """
    example usage: python rank_fusion.py <path to yaml config file>
    """
    parser = argparse.ArgumentParser(description="Rank fusion configuration file")
    parser.add_argument("config", help="Path to the YAML config file")
    return parser.parse_args()

class RankFuser:
    """
    A class that produces a weighted sum of scores from multiple scoring functions,
    then uses those weights to rerank a set of results
    """

    def __init__(self, config: dict[str, float], top_k, rrf_k=60, num_workers=None):
        """
        Initializes the RankFuser with a configuration dictionary that maps scoring function names to their weights.

        Args:
            config (dict[str, float]): metric name: weight
                                       metric names appear as columns in each `results` list
                                       weights are floats, weighting the metric's contribution to the final score
            top_k (int): The number of top results to consider for reranking. If less than number of results, truncates search results to top_k before reranking.
            rrf_k (int): The damping parameter for Reciprocal Rank Fusion (RRF). Default is 60.
        """
        self.config = config
        # self.metrics = list(config.keys())
        # self.weights = list(config.values())
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.num_workers = num_workers if num_workers is not None else min(cpu_count() - 1, 2)

    def _process_row(self, row: dict[str, pd.Series | pd.DataFrame]) -> dict:
        row["results"] = row["results"][: self.top_k]  # Truncate to top_k before reranking
        reranked_results = self._rerank_row(row["results"])
        return compute_stat_matrices(reranked_results)[0] # this fn designed for batch, but we're using it on singles

    def _rerank_row(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Reranks the results DataFrame based on the weighted sum of scores from the configured metrics.

        Args:
            query (pd.Series): The input record for which results are being reranked.
            results (pd.DataFrame): The DataFrame containing results to be reranked.

        Returns:
            pd.DataFrame: The reranked results.
        """
        # Copying already happens in multiprocessing
        # results_df = results.copy()
        results["rrf"] = 0

        # For each metric, compute scores, get ranks, and build weighted RRF
        for score_name, weight in self.config.items():
            ranks = results[score_name].rank(ascending=False, method="first")
            results["rrf"] += weight / (ranks + self.rrf_k)

        # Sort by the weighted score in descending order
        return results.sort_values("rrf", ascending=False).reset_index(drop=True)

    def rerank(self, data: list[dict]) -> list[dict[str, pd.Series | pd.DataFrame]]:
        """
        Expects a list of dictionaries, each containing "record" and "results" keys.
        - "record": the dict representing the original query record
        - "results": ordered list of dicts representing the search results

        Returns:
            list[dict]: The data with reranked results
            dict:
            - "record": pd.Series of the original query record
            - "results": pd.DataFrame of the reranked search results
        """
        with Pool(processes=self.num_workers) as pool:
            reranked_data = list(
                tqdm(
                    pool.imap_unordered(self._process_row, data),
                    total=len(data),
                    desc="Reranking results",
                )
            )
        stat_matrices = {
            "hitrates": np.vstack([item['hitrates'] for item in reranked_data]),
            "ious": np.vstack([item['ious'] for item in reranked_data]),
            "recalls": np.vstack([item['recalls'] for item in reranked_data])
        }
        averages = compute_averages(stat_matrices)
        return averages


def main():
    args = argument_parser()
    path_to_config = args.config
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment_name"]
    search_results_file = config["infile"]
    reranker_config = config["reranker_config"]
    top_k = config["top_k"]
    rrf_k = config.get("rrf_k", 60)
    outdir = config.get("outdir", "experiments/rerankers/results/")

    rank_fuser = RankFuser(
        config=reranker_config,
        top_k=top_k,
        rrf_k=rrf_k,
        num_workers=NUM_WORKERS
    )
