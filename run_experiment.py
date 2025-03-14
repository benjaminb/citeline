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
        description='Run an experiment with specified configuration or build a dataset.')

    # Create mutually exclusive operation groups
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        '--run', action='store_true', help='run the experiment')
    operation_group.add_argument(
        '--build', action='store_true', help='build a dataset')

    # Config argument is optional now
    parser.add_argument('--config', type=str,
                        help='Path to the YAML configuration file.')

    # Dataset building arguments
    parser.add_argument('--num', type=int,
                        help='number of examples to include')
    parser.add_argument('--source', type=str,
                        help='path to source dataset (jsonl)')
    parser.add_argument('--dest', type=str,
                        help='path to destination dataset (jsonl)')
    parser.add_argument('--seed', type=int,
                        help='random seed for dataset sampling')

    args = parser.parse_args()

    # Apply custom validation
    if args.run and not args.config:
        parser.error("--run requires --config")

    if args.build and (not args.num or not args.source or not args.dest):
        parser.error("--build requires --num, --source, and --dest arguments")

    return args


def build_training_dataset(num_examples, source_path, dest_path, seed=None):
    examples = pd.read_json(source_path, lines=True)
    examples = examples.sample(num_examples, random_state=seed)
    examples.to_json(dest_path, orient='records', lines=True)


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
        plt.xlabel(f'Distance Threshold (n = {len(self.dataset)})')
        plt.grid(True)
        plt.savefig(outfile)

    def _get_output_filename_base(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'{self.table}_{self.enrichment}_norm{self.normalize}_{self.metric_to_str[self.metric]}_n{len(self.dataset)}_{current_time}'

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
                    # top_k=2453320
                    top_k=40
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

    def run_batch(self):
        # num_batches = len(self.dataset) // self.batch_size if len(
        #     self.dataset) % self.batch_size == 0 else 1 + len(self.dataset) // self.batch_size
        num_batches = len(self.dataset) // self.batch_size + \
            int(len(self.dataset) % self.batch_size > 0)

        # Grab a batch
        for i in tqdm(range(num_batches), desc="Batches", leave=True):
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

            # TODO: test batch query code here
            # results = self.db.batch_query_vector_table(
            #     table_name=self.table,
            #     query_vectors=embeddings,
            #     metric=self.metric,
            #     top_k=100
            # )
            # print(f"results length: {len(results)}")
            # print(f"First result: {results[0]}")

            # assert len(results) == len(batch) == len(
            #     embeddings), f"Length mismatch: {len(results)} results, {len(batch)} examples, {len(embeddings)} embeddings"

            # for query_results, example in zip(results, batch):

            # for j in range(len(batch)):
            #     example, query_results = batch.iloc[j], results[j]
            #     for threshold in DISTANCE_THRESHOLDS:
            #         predicted_chunks = self._closest_neighbors(
            #             query_results, threshold)
            #         score = self._evaluate_prediction(
            #             example, predicted_chunks)
            #         self.jaccard_scores[threshold].append(score)

            for j in tqdm(range(len(batch)), desc="Querying vectors", leave=True):
                example = batch.iloc[j]
                this_embedding = embeddings[j]
                results = self.db.query_vector_table(
                    table_name=self.table,
                    query_vector=this_embedding,
                    metric=self.metric,
                    use_index=True,
                    # top_k=2453320
                    top_k=1_200_000
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

    if args.build:
        num, source, dest, seed = args.num, args.source, args.dest, args.seed
        print(
            f"Building dataset with {num} examples from {source} to {dest}. Using seed: {seed}")
        build_training_dataset(
            num_examples=num,
            source_path=source,
            dest_path=dest,
            seed=seed
        )
        print("Dataset built.")
        return

    if args.run:
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
        experiment.run_batch()


if __name__ == "__main__":
    main()
