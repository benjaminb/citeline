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
from database.database import DatabaseProcessor
from Enrichers import get_enricher
from embedders import get_embedder

SIMILARITY_THRESHOLDS = np.arange(0, 1.0, 0.01)
REQUIRED_EXPERIMENT_PARAMS = {'dataset', 'table',
                              'metric', 'embedder', 'normalize', 'enrichment'}


def argument_parser():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run an experiment with specified configuration.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to the YAML configuration file.')

    return parser.parse_args()


def read_jsonl(file_path):
    return pd.read_json(file_path, lines=True)


def closest_neighbors(results, threshold: float):
    """
    Assumes that `results` are ordered by similarity, highest to lowest.

    Returns only the results that are above the threshold.
    """
    for i, result in enumerate(results):
        if results[i].similarity < threshold:
            return results[:i]
    return results


def train_test_split_nontrivial(path, split=0.8):
    examples = read_jsonl(path)
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


def get_db_params():
    load_dotenv()
    return {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }


def evaluate_prediction(example, results):
    unique_predicted_dois = set(result.doi for result in results)
    print(f"Number of unique predicted dois: {len(unique_predicted_dois)}")
    citation_dois = set(doi for doi in example.citation_dois)
    score = jaccard_similarity(unique_predicted_dois, citation_dois)
    return score


def jaccard_similarity(set1, set2):
    intersection = np.longdouble(len(set1.intersection(set2)))
    union = np.longdouble(len(set1.union(set2)))
    print(f"Length of intersection: {intersection}")
    print(f"Length of citation_dois: {len(set2)}")
    print(f"Length of predictions: {len(set1)}")
    print(f"Length of union: {union}")
    return intersection / union


def plot_roc_curve(scores: dict, outfile: str):
    plt.figure()
    thresholds = sorted(scores.keys())
    avg_scores = [scores[threshold] for threshold in thresholds]
    plt.plot(thresholds, avg_scores, marker='o',
             linestyle='-', label='Average Jaccard Score')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Average Jaccard Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)  # TODO: add identifying information to filename


def main():
    args = argument_parser()
    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load expermient configs
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    if not REQUIRED_EXPERIMENT_PARAMS.issubset(config.keys()):
        raise ValueError(
            f"Configuration file must contain the following keys: {REQUIRED_EXPERIMENT_PARAMS}. Missing: {REQUIRED_EXPERIMENT_PARAMS - set(config.keys())}")

    # Load database
    db = DatabaseProcessor(get_db_params())
    db.test_connection()
    print(f"Database config: {db.db_params}")

    # Load dataset
    examples = read_jsonl(config['dataset'])

    """
    NOTE: nontrivial_train.index.tolist() will give the line numbers of the original example
    so we can look up the original sentence, etc.
    """

    embedder = get_embedder(
        model_name=config['embedder'],
        device=device,
        normalize=config['normalize']
    )
    print(f"Got embedder: {embedder}")

    # Prepare enricher
    json_files = [
        'data/preprocessed/Astro_Reviews.json',
        'data/preprocessed/Earth_Science_Reviews.json',
        'data/preprocessed/Planetary_Reviews.json'
    ]
    dfs = [pd.read_json(file) for file in json_files]
    data = pd.concat(dfs, ignore_index=True)
    enricher = get_enricher(config['enrichment'], data)

    """
    Experiment procedure
    """
    jaccard_scores = {threshold: [] for threshold in SIMILARITY_THRESHOLDS}

    # Grab a batch
    batch_size = 32
    for i in tqdm(range(1 + len(examples) // batch_size), desc="Batches"):
        batch = examples.iloc[i * batch_size:(i + 1) * batch_size]

        # Enrich sentences
        enriched_examples = enricher.enrich_batch(examples)
        embeddings = embedder(enriched_examples)
        for j in range(len(batch)):
            example = batch.iloc[j]
            this_embedding = embeddings[j]
            results = db.query_vector_table(
                config['table'],
                this_embedding,
                metric=config['metric'],
                top_k=2453320
            )

            for threshold in SIMILARITY_THRESHOLDS:
                predicted_chunks = closest_neighbors(results, threshold)
                score = evaluate_prediction(example, predicted_chunks)
                jaccard_scores[threshold].append(score)

    # Compute average scores
    averages = {round(threshold, 2): float(sum(scores) / len(scores))
                for threshold, scores in jaccard_scores.items()}
    print("Average scores for nontrivial examples:")
    for threshold, score in averages.items():
        print(
            f"Threshold: {threshold}, Average Jaccard: {score}")

    output = {'config': config, 'averages': averages}
    filename_base = f'{config["table"]}_{config["metric"]}_norm{config["normalize"]}_{config["enrichment"]}_{current_time}'
    print(f"Writing results to results.json")
    with open(f'results_{filename_base}.json', 'w') as f:
        json.dump(output, f)

    print("Plotting ROC curve...")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_roc_curve(averages, outfile=f"roc_{filename_base}.png")


if __name__ == "__main__":
    main()
