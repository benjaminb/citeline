import json
from tqdm import tqdm
import os
import pysbd
from concurrent.futures import ProcessPoolExecutor, as_completed

SEG = pysbd.Segmenter(language="en", clean=False)

# Records missing any of these keys are excluded from the dataset
REQUIRED_KEYS = {'title', 'body', 'abstract', 'doi', 'reference', 'bibcode'}


def load_dataset(path):
    with open(path, 'r') as file:
        data = json.load(file)

    total_records = len(data)
    data = [d for d in data if REQUIRED_KEYS.issubset(d.keys())]
    complete_records = len(data)
    print(f"{path}: {complete_records}/{total_records} have all required keys")

    for record in data:
        record['title'] = record['title'][0]

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


def main():
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


if __name__ == "__main__":
    main()
