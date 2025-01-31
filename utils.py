import json


def load_dataset(path):
    with open(path, 'r') as file:
        data = json.load(file)

    for record in data['metadatas']:
        record['reference'] = json.loads(record['reference'])
        record['doi'] = json.loads(record['doi'])

    return data['metadatas']
