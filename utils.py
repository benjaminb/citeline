import json

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
