import argparse
import json
import os
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from citeline.statistics import compute_averages, compute_individual_stats

load_dotenv('.env')
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

    def __init__(self, infile: str, config: dict[str, float], top_k, rrf_k=60, num_workers=None):
        """
        Initializes the RankFuser with a configuration dictionary that maps scoring function names to their weights.

        Args:
            config (dict[str, float]): metric name: weight
                                       metric names appear as columns in each `results` list
                                       weights are floats, weighting the metric's contribution to the final score
            top_k (int): The number of top results to consider for reranking. If less than number of results, truncates search results to top_k before reranking.
            rrf_k (int): The damping parameter for Reciprocal Rank Fusion (RRF). Default is 60.
        """
        self.infile = infile
        self.config = config
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.num_workers = num_workers if num_workers is not None else min(cpu_count() - 1, 2)

    def _process_row(self, row: dict[str, pd.Series | pd.DataFrame]) -> dict:
        # Convert to pd
        example = pd.Series(row["record"])
        results = pd.DataFrame(row["results"])

        # Process search results and compute stats
        results = results[: self.top_k]  # Truncate to top_k before reranking
        results = self._rerank_row(results)
        return compute_individual_stats(example, results)

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

    def _read_row_generator(self):
        """Generator that yields rows from the input file."""
        with open(self.infile, 'r') as f:
            for line in f:
                yield json.loads(line)

    def _serialize_averages(self, averages: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """Convert numpy arrays to lists for JSON serialization."""
        return {key: value.tolist() for key, value in averages.items()}

    def execute(self) -> dict[str, np.ndarray]:
        """
        Expects a list of dictionaries, each containing "record" and "results" keys.
        - "record": the dict representing the original query record
        - "results": ordered list of dicts representing the search results

        Returns average stats@k:
        {
            "hitrate": np.ndarray,
            "iou": np.ndarray,
            "recall": np.ndarray
        }
        """

        with Pool(processes=self.num_workers) as pool:
            reranked_stats = list(
                tqdm(
                    pool.imap_unordered(self._process_row, self._read_row_generator()),
                    desc="Reranking results",
                )
            )
        stat_matrices = {
            "hitrates": np.vstack([item['hitrate'] for item in reranked_stats]),
            "ious": np.vstack([item['iou'] for item in reranked_stats]),
            "recalls": np.vstack([item['recall'] for item in reranked_stats])
        }
        averages_np = compute_averages(stat_matrices)
        return self._serialize_averages(averages_np)


def main():
    args = argument_parser()
    path_to_config = args.config
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    fusion_name = config["name"]
    infile_path = config["infile"]
    reranker_config = config["reranker_config"]
    top_k = config["top_k"]
    rrf_k = config.get("rrf_k", 60)
    outdir = config.get("outdir", "experiments/rerankers/results/")

    rank_fuser = RankFuser(
        infile=infile_path,
        config=reranker_config,
        top_k=top_k,
        rrf_k=rrf_k,
        num_workers=NUM_WORKERS
    )

    average_stats = rank_fuser.execute()
    log = {'config': config} | average_stats
    # Make a dir concatenating the name and the timestamp
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{fusion_name}_rank_fusion_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"Results saved to {outdir}/{fusion_name}_rank_fusion_log.json")

if __name__ == "__main__":
    main()