import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing import Pool

from citeline.rank_fuser import RankFuser
from citeline.statistics import compute_stat_matrices, compute_averages
from citeline.helpers.plot import plot_stats_at_k


CHUNK_SIZE = 1000

# Global variable for each worker process
_rank_fuser = None


def init_worker(config, k):
    """Initialize the rank fuser once per worker process."""
    global _rank_fuser
    _rank_fuser = RankFuser(config, k=k)


def process_batch(batch):
    """Process a single batch using the pre-initialized rank fuser."""
    global _rank_fuser
    batch_reranked = _rank_fuser.rerank(batch)
    return compute_stat_matrices(batch_reranked)


def yield_chunk(filepath, chunk_size):
    """Yield lists of parsed JSON objects from a JSONL file in chunks."""
    chunk = []
    with open(filepath, "r") as f:
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def get_line_count(filepath: str) -> int:
    assert filepath.endswith(".jsonl"), "get_line_count: File must be a .jsonl file"
    with open(filepath, "r") as f:
        return sum(1 for _ in f)


class RerankExperiment:
    def __init__(
        self,
        experiment_name,
        search_results_file,
        reranker_config: dict,
        k=60,
        outdir="experiments/rerankers/results/",
        num_workers=12,
    ):
        self.experiment_name = experiment_name
        self.search_results_file = search_results_file
        self.reranker_config = reranker_config
        self.k = k
        self.num_workers = num_workers
        self.outdir = os.path.join(outdir, experiment_name)
        os.makedirs(self.outdir, exist_ok=True)

    def run(self):
        """Run the reranking experiment with multiprocessing."""
        # Count lines in the file
        num_lines = get_line_count(self.search_results_file)

        # Generate batches
        batches = yield_chunk(self.search_results_file, CHUNK_SIZE)

        # Process batches in parallel
        all_chunk_stats = []
        with Pool(self.num_workers, initializer=init_worker, initargs=(self.reranker_config, self.k)) as pool:
            # chunksize controls how many tasks are sent to workers at once
            # This allows I/O to happen while CPU work is being done
            for chunk_stats in tqdm(
                pool.imap_unordered(process_batch, batches, chunksize=2),
                desc="Processing chunks",
                total=(num_lines + CHUNK_SIZE - 1) // CHUNK_SIZE,
            ):
                all_chunk_stats.append(chunk_stats)

            pool.close()
            pool.join()

        # Combine all chunks
        stats = None
        for chunk_stats in all_chunk_stats:
            if stats is None:
                stats = {k: v for k, v in chunk_stats.items()}
            else:
                for k in stats:
                    stats[k] = np.vstack([stats[k], chunk_stats[k]])

        average_stats = compute_averages(stats)

        # Save stats
        data = {
            "config": self.reranker_config,
            "k": self.k,
            "average_stats": {k: v.tolist() for k, v in average_stats.items()},
        }
        with open(os.path.join(self.outdir, "reranked_stats.json"), "w") as f:
            json.dump(data, f, indent=2)

        # Make & save plot
        plot_stats_at_k(
            average_stats,
            output_path=os.path.join(self.outdir, "reranked_stats_at_k.png"),
            k_cutoff=400,
            title=f"{self.experiment_name}: Reranked Search Results Stats@k",
        )

        return average_stats


def main():
    NUM_WORKERS = 11
    search_results_file = "experiments/multiple_query_expansion/results/search_results.jsonl"
    for bm25_weight in np.arange(0.0, 1.1, 0.1):
        experiment_name = f"rrf_reciprocal_rank_bm25_{bm25_weight:.1f}"
        reciprocal_rank_weight = 1.0 - bm25_weight
        config = {
            "bm25_scratch": bm25_weight,
            "reciprocal_rank": reciprocal_rank_weight,
        }
        experiment = RerankExperiment(
            experiment_name=experiment_name,
            search_results_file=search_results_file,
            reranker_config=config,
            k=60,
            num_workers=NUM_WORKERS,
        )
        experiment.run()

    # experiment_name = "bm25_scratch_0.42_position_0.58"
    # config = {"bm25_scratch": 0.42, "position": 0.58}
    # search_results_file = "experiments/multiple_query_expansion/results/search_results.jsonl"

    # experiment = RerankExperiment(
    #     experiment_name=experiment_name,
    #     search_results_file=search_results_file,
    #     reranker_config=config,
    #     k=60,
    #     num_workers=NUM_WORKERS,
    # )

    # experiment.run()


if __name__ == "__main__":
    main()
