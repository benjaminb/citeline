import argparse
import json
from tqdm import tqdm
import os
import pandas as pd
import pysbd
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
Originally, data was provided in various JSON files representing papers from various fields or journals.
These records contained the following keys (except for a few missing certain keys):
  'bibcode', 'abstract', 'aff', 'author', 'bibstem', 'doctype', 'doi',
  'id', 'keyword', 'pubdate', 'title', 'read_count', 'reference', 'data',
  'citation_count', 'citation', 'body', 'dois', 'loaded_from'

To make the data uniform, we drop any records missing any of the REQUIRED_KEYS. In addition, this code
  - Extracts the first doi from each record and stores it in a new key 'doi' (the old 'doi' key is renamed to 'dois', and is a list).
  - For research data (the reference dataset): drop any duplicates
  - For review data (the query dataset): segment the body text into sentences and merge any short sentences (less than 60 characters).
The final output is written to two separate JSONL files: 'research.jsonl' and 'reviews.jsonl'.
"""

SEG = pysbd.Segmenter(language="en", clean=False)

# Records missing any of these keys are excluded from the dataset
REQUIRED_KEYS = {'title', 'body', 'abstract', 'doi', 'reference', 'bibcode'}


def argument_parser():
    parser = argparse.ArgumentParser(description="Preprocess datasets.")

    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        '--research', action='store_true', help="Process research datasets")
    operation_group.add_argument(
        '--reviews', action='store_true', help="Process review datasets")

    args = parser.parse_args()
    return args


def load_dataset(path):
    with open(path, 'r') as file:
        data = json.load(file)

    total_records = len(data)
    data = [d for d in data if REQUIRED_KEYS.issubset(d.keys())]
    complete_records = len(data)
    print(f"{path}: {complete_records}/{total_records} have all required keys")

    for record in data:
        record['title'] = record['title'][0]

        # Extract first DOI in list as 'doi'
        assert isinstance(
            record['doi'], list), f"DOI expected to be a list, but it was {type(record['doi'])}, value: {record['doi']}"
        record['dois'] = record['doi']
        record['doi'] = record['doi'][0]
        record['loaded_from'] = path
    return data


def merge_short_sentences(sentences, threshold=60):
    """
    Returns a list of sentences where sentences below the threshold length
    are re-concatenated with the following sentence. If the result is still 
    below the threshold length, the process is repeated until the threshold
    is reached.
    """
    merged_sentences = []
    for i in range(len(sentences) - 1):
        if len(sentences[i]) < threshold:
            sentences[i + 1] = sentences[i] + sentences[i + 1]
        else:
            merged_sentences.append(sentences[i])

    # Handle the last sentence
    if len(sentences[-1]) < threshold or len(merged_sentences) > 0:
        merged_sentences[-1] = merged_sentences[-1] + sentences[-1]
    else:
        merged_sentences.append(sentences[-1])
    return merged_sentences


def process_record(record):
    sentences = SEG.segment(record['body'])
    record['body_sentences'] = merge_short_sentences(sentences)
    return record


def get_unique_records(datasets: list[str]) -> pd.DataFrame:
    """
    This function loads multiple datasets, concatenates them into a single DataFrame,
    and drops duplicates based on the 'doi' field.
    """
    records = [
        record for dataset in datasets for record in load_dataset(dataset)]
    df = pd.DataFrame(records)
    full_length = len(df)
    df = df.drop_duplicates(subset=['doi'])
    unique_length = len(df)

    print(
        f"Loaded {full_length} records, {unique_length} unique records based on 'doi'")
    return df


def write_review_data(datasets):
    """
    This function processes review datasets and writes the output to a single JSON file.
    Each record is processed to segment the body into sentences and merge short sentences.
    """
    df = get_unique_records(datasets)
    records = df.to_dict('records')

    # Process records in parallel
    results = []
    max_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing jobs
        futures = [executor.submit(process_record, record)
                   for record in records]

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing records"):
            results.append(future.result())

    # Convert results back to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_json('data/preprocessed/reviews.jsonl',
                      orient='records', lines=True)


def write_research_data(datasets):
    records = [record for dataset in datasets for record in load_dataset(
        'data/json/' + dataset)]
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['doi'])
    df.to_json('data/preprocessed/research.jsonl',
               orient='records', lines=True)


def preprocess_data(datasets: list[str], output_file: str):
    """
    This function processes review datasets and writes the output to a single JSON file.
    Each record is processed to segment the body into sentences and merge short sentences.
    """
    df = get_unique_records(datasets)
    records = df.to_dict('records')

    # Process records in parallel
    results = []
    max_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing jobs
        futures = [executor.submit(process_record, record)
                   for record in records]

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing records"):
            results.append(future.result())

    # Convert results back to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_json(output_file, orient='records', lines=True)


def main():
    args = argument_parser()

    if args.reviews:
        """
        * Loads the json files for the review datasets, makes sure we only retain unique records (based on doi),
        * Takes each record body text, segments it into sentences, and merges any short sentences (less than 60 characters)
        * Places the sentence segments into a new field called 'body_sentences'.
        * Writes the final output to a single jsonl file called 'reviews.jsonl'.
        """
        write_review_data(
            ['Astro_Reviews.json', 'Earth_Science_Reviews.json', 'Planetary_Reviews.json'])

    if args.research:
        """
        In addition to loading each research dataset and preprocessing the individual records,
        this branch also drops any duplicate records based on the 'doi' field and writes the final
        output to a single jsonl file called 'research.json'.
        """
        datasets = ['data/json/Astro_Research.json', 'data/json/Earth_Science_Research.json',
                    'data/json/Planetary_Research.json', 'data/json/doi_articles.json', 'data/json/salvaged_articles.json']
        preprocess_data(
            datasets, output_file='data/preprocessed/research.jsonl')
        return


if __name__ == "__main__":
    main()
