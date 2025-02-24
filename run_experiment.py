"""
To run an experiment:

database: 'test'
table: 'test_bge'


specify 
- the database
- the database vector table to use (this sets the embedding model, normalization used or not)
- the distance metric to use, cosine, l2, or ip
- the % of trivial examples relative to nontrivial examples to use in the training set
- the method you'll use to augment each sentence in the training set
    -e.g. previous n sentences, title + abstract, etc.



user specifies a vector table in the database they want to run the experiment on, plus a distance metric

Then we load the training set into memory from dataset/no_reviews/
    - load all nontrivial examples
    - load a portion of trivial examples
    
set up separate lists for trivial and nontrivial examples' IoU scores
For each example in training set, get the `sent_no_cit` value from the example
    - do this in batches
    - use the specified augmentation function to enrich the sent_no_cit
    - embed the enriched sentence
    - get the top k nearest neighbors, particularly their doi's
    - compute intersection over union 
    - log the predicted top doi's for the sentence
        - log trivial and nontrivial examples separately

After all examples are processed, compute the average intersection over union for the training set
    - average IoU for trivial examples
    - average IoU for nontrivial examples
    - average IoU for all examples

Write out results to file


"""
import argparse
import json
import os
import torch
import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from time import time
from database.database import DatabaseProcessor
from embedding_functions import get_embedding_fn

SIMILARITY_THRESHOLDS = np.arange(0, 1.0, 0.1)


def argument_parser():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run an experiment with specified configuration.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to the YAML configuration file.')

    return parser.parse_args()


def read_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

# cut off results below threshold similarity


def closest_neighbors(results, threshold: float):
    """
    Assumes that `results` are ordered by similarity, highest to lowest.

    Returns only the results that are above the threshold.
    """
    for i, result in enumerate(results):
        if results[i].similarity < threshold:
            return results[:i]
    return results


def train_test_split(trivial_proportion=1.0, split=0.8):
    nontrivial_examples = read_jsonl(
        'data/dataset/no_reviews/nontrivial.jsonl')
    trivial_examples = read_jsonl('data/dataset/no_reviews/trivial.jsonl')

    # Select a proportion of trivial examples to use
    num_trivial_to_sample = min(len(trivial_examples), int(
        len(nontrivial_examples) * trivial_proportion))
    trivial_examples = trivial_examples.sample(
        num_trivial_to_sample, random_state=42)

    # Select 80% of nontrivial examples for training
    nontrivial_train = nontrivial_examples.sample(frac=split, random_state=42)
    nontrivial_test = nontrivial_examples.drop(nontrivial_train.index)

    # Select 80% of trivial examples for training
    trivial_train = trivial_examples.sample(frac=split, random_state=42)
    trivial_test = trivial_examples.drop(trivial_train.index)

    print(
        f"Selected {len(nontrivial_train)} nontrivial examples for training out of {len(nontrivial_examples)}")
    print(
        f"Selected {len(trivial_train)} trivial examples for training out of {len(trivial_examples)}")

    return nontrivial_train, nontrivial_test, trivial_train, trivial_test


def get_db_params(config):
    load_dotenv()
    return {
        'dbname': config['database'],
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }


def evaluate_prediction(example, results):
    unique_predicted_dois = set(result.doi for result in results)
    print(f"Number of unique predicted dois: {len(unique_predicted_dois)}")
    citation_dois = set(doi for doi in example['citation_dois'])
    score = jaccard_similarity(unique_predicted_dois, citation_dois)
    return score


def jaccard_similarity(set1, set2):
    intersection = np.longdouble(len(set1.intersection(set2)))
    print(f"Length of intersection: {intersection}")
    union = np.longdouble(len(set1.union(set2)))
    print(f"Length of citation_dois: {len(set2)}")
    print(f"Length of predictions: {len(set1)}")
    print(f"Length of union: {union}")
    return intersection / union


def main():
    args = argument_parser()
    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    processor = DatabaseProcessor(get_db_params(config))
    processor.test_connection()

    nontrivial_train, nontrivial_test, trivial_train, trivial_test = train_test_split()
    """
    NOTE: nontrivial_train.index.tolist() will give the line numbers of the original example
    so we can look up the original sentence, etc.
    """

    embedder = get_embedding_fn(
        model_name=config['embedder'],
        device=device,
        normalize=config['normalize']
    )

    # nontrivial_jaccard_scores, trivial_jaccard_scores = [], []

    nontrivial_jaccard_scores = {threshold: []
                                 for threshold in SIMILARITY_THRESHOLDS}
    batch_size = 16
    # # Iterate over nontrivial examples in batches
    num_batches = len(nontrivial_train) // batch_size
    start_time = time()
    for i in range(1):
        start, end = i * batch_size, (i + 1) * batch_size
        batch = nontrivial_train.iloc[start:end]
        sentences = batch['sent_no_cit'].tolist()

        # enrich the sentence

        # embed the enriched sentence
        embeddings = embedder(sentences)
        for (idx, row), embedding in zip(batch.iterrows(), embeddings):
            example = row.to_dict()
            results = processor.query_vector_table(
                config['table'],
                embedding,
                metric=config['metric'],
                top_k=2453320)

            # NOTE: remove the slice on similarity thresholds
            for threshold in SIMILARITY_THRESHOLDS[:2]:
                predicted_chunks = closest_neighbors(results, threshold)
                score = evaluate_prediction(example, predicted_chunks)
                nontrivial_jaccard_scores[threshold].append(score)

        print('====')

    total_time = time() - start_time
    print("Number of total scores:")
    for threshold, scores in nontrivial_jaccard_scores.items():
        print(f"Threshold: {threshold}, Number of scores: {len(scores)}")

    averages = {round(threshold, 1): sum(scores) / len(scores)
                for threshold, scores in nontrivial_jaccard_scores.items()}
    with open('results.json', 'w') as f:
        json.dump(averages, f)

    print("Average scores for nontrivial examples:")
    for threshold, score in averages.items():
        print(
            f"Threshold: {threshold}, Average Jaccard: {np.frombuffer(score)}")

    # log trivial and nontrivial examples separately


if __name__ == "__main__":
    main()
