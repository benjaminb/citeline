import json
from tqdm import tqdm

REQUIRED_KEYS = {'title', 'body', 'abstract', 'doi', 'reference', 'bibcode'}


def load_dataset(path):
    with open(path, 'r') as file:
        data = json.load(file)

    total_records = len(data)
    data = [d for d in data if REQUIRED_KEYS.issubset(d.keys())]
    complete_records = len(data)
    print(f"{path}: {complete_records}/{total_records} have all required keys")

    for record in data:
        record['title'] = ': '.join(record['title'])

    return data


def write_incomplete_data(path, outfile):
    with open(path, 'r') as file:
        data = json.load(file)

    incompletes = []
    for i, record in tqdm(enumerate(data)):
        missing_keys = REQUIRED_KEYS - record.keys()
        if missing_keys:
            incompletes.append(
                (i, record.get('doi', 'unk'), '|'.join(missing_keys)))

    # write out to csv
    with open(outfile, 'w') as file:
        for i, doi, keys in incompletes:
            file.write(f"{i},{doi},{keys}\n")
