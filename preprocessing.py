import argparse
import json
from tqdm import tqdm
import os
import pandas as pd
import pysbd
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    if len(sentences[-1]) < threshold or len(merged_sentences[-1]) < threshold:
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
    records = [record for dataset in datasets for record in load_dataset(
        'data/json/' + dataset)]
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['doi'])
    return df

def write_review_data(datasets):
    """
    This function processes review datasets and writes the output to a single JSON file.
    Each record is processed to segment the body into sentences and merge short sentences.
    """
    df = get_unique_records(datasets)

    # On the df, apply the process_record function to each row
    df['body_sentences'] = df['body'].apply(lambda x: merge_short_sentences(SEG.segment(x)))
    df.to_json('data/preprocessed/reviews.jsonl', orient='records', lines=True)



    with open('data/preprocessed/reviews.json', 'w') as f:
        f.write('[')  # Start the JSON array
        for dataset in datasets:
            records = load_dataset('data/json/' + dataset)
            with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
                futures = [executor.submit(process_record, record)
                           for record in records]
                results = []
                for future in tqdm(as_completed(futures), total=len(futures)):
                    results.append(future.result())
                for i, record in enumerate(results):
                    f.write(json.dumps(record))
                    if i < len(records) - 1:
                        f.write(',')  # Add a comma between JSON objects
        f.write(']')  # End the JSON array

def write_research_data(datasets):
    records = [record for dataset in datasets for record in load_dataset(
        'data/json/' + dataset)]
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['doi'])
    df.to_json('data/preprocessed/research.jsonl', orient='records', lines=True)

def main():
    args = argument_parser()

    if args.reviews:
        for dataset in ['Astro_Reviews.json', 'Earth_Science_Reviews.json', 'Planetary_Reviews.json']:
            print(f"Processing {dataset}...")
            records = load_dataset('data/json/' + dataset)

            with open('data/preprocessed/' + dataset, 'w') as f:
                f.write('[')  # Start the JSON array
                with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                    futures = [executor.submit(process_record, record)
                               for record in records]
                    results = []
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        results.append(future.result())
                    for i, record in enumerate(results):
                        f.write(json.dumps(record))
                        if i < len(records) - 1:
                            f.write(',')  # Add a comma between JSON objects
                f.write(']')  # End the JSON array

    if args.research:
        """
        In addition to loading each research dataset and preprocessing the individual records,
        this branch also drops any duplicate records based on the 'doi' field and writes the final
        output to a single jsonl file called 'research.json'.
        """
        datasets = ['Astro_Research.json', 'Earth_Science_Research.json',
                    'Planetary_Research.json', 'doi_articles.json', 'salvaged_articles.json']
        write_research_data(datasets)
        return

    if args.reviews:
        datasets = ['Astro_Reviews.json',
                    'Earth_Science_Reviews.json', 'Planetary_Reviews.json']


if __name__ == "__main__":
    main()
