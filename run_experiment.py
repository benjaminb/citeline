import argparse
import json
import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from database.database import DatabaseProcessor, get_db_params
from Enrichers import get_enricher
from Embedders import get_embedder

DISTANCE_THRESHOLDS = np.arange(1.0, 0.0, -0.01)


def argument_parser():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run an experiment with specified configuration.')
    parser.add_argument('config', type=str,
                        help='Path to the YAML configuration file.')

    return parser.parse_args()


def train_test_split_nontrivial(path, split=0.8):
    examples = pd.read_json(path, lines=True)
    train = examples.sample(frac=split, random_state=42)
    test = examples.drop(train.index)
    return train, test

# def train_test_split(trivial_proportion=1.0, split=0.8):
#     nontrivial_examples = read_jsonl(
#         'data/dataset/no_reviews/nontrivial.jsonl')
#     trivial_examples = read_jsonl('data/dataset/no_reviews/trivial.jsonl')

#     # Select a proportion of trivial examples to use
#     num_trivial_to_sample = min(len(trivial_examples), int(
#         len(nontrivial_examples) * trivial_proportion))
#     trivial_examples = trivial_examples.sample(
#         num_trivial_to_sample, random_state=42)

#     # Select 80% of nontrivial examples for training
#     nontrivial_train = nontrivial_examples.sample(frac=split, random_state=42)
#     nontrivial_test = nontrivial_examples.drop(nontrivial_train.index)

#     # Select 80% of trivial examples for training
#     trivial_train = trivial_examples.sample(frac=split, random_state=42)
#     trivial_test = trivial_examples.drop(trivial_train.index)

#     print(
#         f"Selected {len(nontrivial_train)} nontrivial examples for training out of {len(nontrivial_examples)}")
#     print(
#         f"Selected {len(trivial_train)} trivial examples for training out of {len(trivial_examples)}")

#     return nontrivial_train, nontrivial_test, trivial_train, trivial_test


class Experiment:
    metric_to_str = {
        'vector_l2_ops': 'L2',
        'vector_cosine_ops': 'cosine',
        'vector_ip_ops': 'ip'
    }

    def __init__(self, device: str, dataset_path: str, table: str, metric: str, embedding_model_name: str, normalize: bool, enrichment: str, batch_size: int = 16):
        # Set up configs
        self.device = device

        """
        NOTE: nontrivial_train.index.tolist() will give the line numbers of the original example
        so we can look up the original sentence, etc.
        """
        self.dataset = pd.read_json(dataset_path, lines=True)
        self.dataset_path = dataset_path

        self.table = table
        self.metric = metric
        self.batch_size = batch_size
        self.embedder = get_embedder(
            model_name=embedding_model_name,
            device=device,
            normalize=normalize
        )
        self.normalize = normalize
        self.enrichment = enrichment
        self.enricher = get_enricher(enrichment)

        # Initialize database
        self.db = DatabaseProcessor(get_db_params())
        self.db.test_connection()

        # Prepare attributes for results
        self.jaccard_scores = {threshold: []
                               for threshold in DISTANCE_THRESHOLDS}
        """
        Dictionary of average Jaccard scores for each distance threshold
        {0.5: 0.1785} means after only keeping query results with distance < 0.5, the average IoU score for
        all examples in the dataset is 0.1785
        """
        self.averages = {}

    def _closest_neighbors(self, results, threshold: float):
        """
        Assumes that `results` are ordered by distance, lowest to highest.

        Returns only the results that have distance below the threshold
        """
        for i in range(len(results)):
            if results[i].distance > threshold:
                return results[:i]
        return results

    def _evaluate_prediction(self, example, results):
        unique_predicted_dois = set(result.doi for result in results)
        citation_dois = set(doi for doi in example.citation_dois)
        score = self._jaccard_similarity(unique_predicted_dois, citation_dois)
        return score

    def _jaccard_similarity(self, set1, set2):
        intersection = np.longdouble(len(set1.intersection(set2)))
        union = np.longdouble(len(set1.union(set2)))
        return intersection / union

    def _plot_roc_curve(self, filename_base: str):
        outfile = f"experiments/results/{filename_base}/roc_{filename_base}.png"

        plt.figure()
        thresholds = sorted(self.averages.keys())
        avg_scores = [self.averages[threshold] for threshold in thresholds]
        plt.plot(thresholds, avg_scores, marker='.',
                 linestyle='-', label='Average Jaccard Score')
        plt.xlabel('Distance Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(outfile)

    def _get_output_filename_base(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'{self.table}_{self.enrichment}_norm{self.normalize}_{self.metric_to_str[self.metric]}_{current_time}'

    def _write_json_results(self, filename_base):
        # Prep results and outfile name
        output = {'config': self.get_config_dict(), 'averages': self.averages}

        # Create directory if it doesn't exist
        if not os.path.exists(f'experiments/results/{filename_base}'):
            os.makedirs(f'experiments/results/{filename_base}')

        # Write and plot results
        with open(f'experiments/results/{filename_base}/results_{filename_base}.json', 'w') as f:
            json.dump(output, f)

    def _write_results(self):
        # Prep results and outfile name
        metric_str = self.metric_to_str.get(self.metric)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f'{self.table}_{self.enrichment}_norm{self.normalize}_{metric_str}_{current_time}'

        # Create directory if it doesn't exist
        if not os.path.exists(f'experiments/results/{filename_base}'):
            os.makedirs(f'experiments/results/{filename_base}')

        # Write and plot results
        self._write_json_results(filename_base)
        self._plot_roc_curve(filename_base)

    def get_config_dict(self):
        return {
            'dataset': self.dataset_path,
            'table': self.table,
            'metric': self.metric,
            'embedder': self.embedder.model_name,
            'normalize': self.normalize,
            'enrichment': self.enrichment,
            'batch_size': self.batch_size
        }

    def run(self):
        # Compute number of batches. If the dataset size is not a multiple of the batch size, add one more batch
        num_iterations = len(self.dataset) // self.batch_size if len(
            self.dataset) % self.batch_size == 0 else 1 + len(self.dataset) // self.batch_size
        
        # Grab a batch
        for i in tqdm(range(num_iterations), desc="Batches", leave=True):
            if i % 50 == 0:
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()

            batch = self.dataset.iloc[i *
                                      self.batch_size:(i + 1) * self.batch_size]

            # Enrich and embed batch
            enriched_batch = self.enricher.enrich_batch(batch)
            embeddings = self.embedder(enriched_batch)

            for j in tqdm(range(len(batch)), desc="Querying vectors", leave=True):
                example = batch.iloc[j]
                this_embedding = embeddings[j]
                results = self.db.query_vector_table(
                    table_name=self.table,
                    query_vector=this_embedding,
                    metric=self.metric,
                    top_k=2453320
                )

                for threshold in DISTANCE_THRESHOLDS:
                    predicted_chunks = self._closest_neighbors(
                        results, threshold)
                    score = self._evaluate_prediction(
                        example, predicted_chunks)
                    self.jaccard_scores[threshold].append(score)

        # Compute average scores and write out results
        self.averages = {round(threshold, 2): float(sum(scores) / len(scores))
                         for threshold, scores in self.jaccard_scores.items()}
        self._write_results()


def main():
    args = argument_parser()

    # Load expermient configs
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.mps.is_available() else 'cpu'

    # Set up and run experiment
    experiment = Experiment(
        device=device,
        dataset_path=config['dataset'],
        table=config['table'],
        metric=config['metric'],
        embedding_model_name=config['embedder'],
        normalize=config['normalize'],
        enrichment=config['enrichment'],
        batch_size=config.get('batch_size', 16)
    )
    experiment.run()


if __name__ == "__main__":
    main()
