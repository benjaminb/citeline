import json
import requests
from tqdm import tqdm

REQUIRED_KEYS = {'title', 'body', 'abstract', 'doi', 'reference', 'bibcode'}

# NOTE: moved to preprocessing.py
# def load_dataset(path):

def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    result = response.json()
    return result['response']


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
