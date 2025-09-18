import json
import requests
from tqdm import tqdm

REQUIRED_KEYS = {"title", "body", "abstract", "doi", "reference", "bibcode"}

# NOTE: moved to preprocessing.py
# def load_dataset(path):


def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "deepseek-r1", "prompt": prompt, "stream": False}

    response = requests.post(url, json=payload)
    result = response.json()
    return result["response"]
