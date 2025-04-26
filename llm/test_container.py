import requests

import requests
import json

url = "http://localhost:11434/api/generate"
data = {
    "model": "llama3.2:1b",  # Replace with your model name
    "prompt": "Write me a story about elves meeting an AI",
    "stream": False,  # Set True for streaming responses
}

response = requests.post(url, json=data)
if response.status_code == 200:
    print(response.json()["response"])
else:
    print(f"Error {response.status_code}: {response.text}")
