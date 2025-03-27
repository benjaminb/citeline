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
REQUIRED_KEYS = {'title', 'body', 'abstract',
                 'doi', 'reference', 'bibcode',
                 # 'keyword'
                 }


def argument_parser():
    parser = argparse.ArgumentParser(description="Preprocess datasets.")

    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        '--research', action='store_true', help="Process research datasets")
    operation_group.add_argument(
        '--reviews', action='store_true', help="Process review datasets")
    # Add a list argument 'datasets'

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

        # Additional keys
        record['loaded_from'] = path
        record['body_sentences'] = []
    return data


def merge_short_sentences(sentences, threshold=60):
    """
    Returns a list of sentences where sentences below the threshold length
    are concatenated with the following sentence until they exceed the threshold.
    """
    merged_sentences = []
    buffer = ""

    for sentence in sentences:
        if len(buffer) + len(sentence) < threshold:
            buffer += sentence + " "  # Keep accumulating
        else:
            if buffer:
                merged_sentences.append(buffer.strip())  # Commit the buffer
            buffer = sentence  # Reset buffer to current sentence

    if buffer:  # Add any remaining buffer at the end
        merged_sentences.append(buffer.strip())

    return merged_sentences


def process_record(record):
    """
    This function processes a single record by segmenting the body text into sentences, then merges
    short sentences using the merge_short_sentences function. The final output is a record with a new
    key 'body_sentences' containing the segmented and merged sentences.
    """
    sentences = SEG.segment(record['body'])
    # If there are no sentences in the body, log the error and return the record with blank 'body_sentences' key
    if not sentences:
        print(f"Empty sentences for record: {record['doi']}")
        with open('empty_sentences.csv', 'a') as file:
            file.write(f"{record['doi']},{record['title']}\n")
        return record

    # Typical case: we have sentences so merge the short ones
    record['body_sentences'] = merge_short_sentences(sentences)
    return record


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes review datasets and writes the output to a single JSON file.
    Each record is processed to segment the body into sentences and merge short sentences.
    """
    # Replace invalid days and months (12-00-00 to 12-01-01)
    df['pubdate'] = df['pubdate'].str.replace(r'-00', '-01', regex=True)
    records = df.to_dict('records')

    # Process records in parallel
    results = []
    max_workers = max(1, os.cpu_count() - 2)
    print(f"Processing records with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing jobs
        futures = [executor.submit(process_record, record)
                   for record in records]

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing records"):
            results.append(future.result())

    return pd.DataFrame(results)


def write_data(datasets, output_file: str, deduplicate_from=None):
    # Get all records
    records = [record for dataset in datasets for record in load_dataset(
        'data/json/' + dataset)]

    # Drop duplicates based on 'doi'
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['doi'])

    # Drop any records with 'body' less than 5000 characters
    df = df[df['body'].str.len() >= 5000]

    # Drop any records that are also in the review dataset
    if deduplicate_from:
        other_dataset = pd.read_json(deduplicate_from, lines=True)
        other_dois = set(other_dataset['doi'])
        df = df[~df['doi'].isin(other_dois)]

    # Preprocess records and write output
    df = preprocess_data(df)
    df.to_json(output_file, orient='records', lines=True)


def main():
    args = argument_parser()

    if args.reviews:
        """
        * Loads the json files for the review datasets, makes sure we only retain unique records (based on doi),
        * Takes each record body text, segments it into sentences, and merges any short sentences (less than 60 characters)
        * Places the sentence segments into a new field called 'body_sentences'.
        * Writes the final output to a single jsonl file called 'reviews.jsonl'.
        """
        write_data(
            # datasets=['Astro_Reviews.json'],
            datasets=['Astro_Reviews.json',
                      'Earth_Science_Reviews.json', 'Planetary_Reviews.json'],
            output_file='data/testdate.jsonl',
            deduplicate_from=None
        )
        return

    if args.research:
        """
        In addition to loading each research dataset and preprocessing the individual records,
        this branch also drops any duplicate records based on the 'doi' field and writes the final
        output to a single jsonl file called 'research.json'.

        NOTE: review data should be written out first to ensure the research data doesn't have review paper records in it
        """
        datasets = ['Astro_Research.json', 'Earth_Science_Research.json',
                    'Planetary_Research.json', 'doi_articles.json', 'salvaged_articles.json']
        write_data(
            datasets=datasets,
            output_file='data/preprocessed/research.jsonl',
            deduplicate_from='data/preprocessed/reviews.jsonl')
        return


if __name__ == "__main__":
    main()
