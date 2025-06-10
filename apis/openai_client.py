"""
This creates an OpenAI client which also can be used for DeepSeek
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file from project root
load_dotenv("../.env")


def deepseek_client():
    assert "DEEPSEEK_API_KEY" in os.environ, "DEEPSEEK_API_KEY must be set in environment variables"
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    return client


def deepseek_chat(prompt: str, vars: dict = None, response_model=None):
    client = deepseek_client()
    response = client.responses.parse(model="deepseek-chat", input="")


def main():
    assert "DEEPSEEK_API_KEY" in os.environ, "DEEPSEEK_API_KEY must be set in environment variables"
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key="api_key", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        stream=False,
    )
    print(response.choices[0].message.content)

    from pprint import pprint

    print("Response object:")
    pprint(response)

    import json

    with open("deepseek_response_object.json", "w") as f:
        json.dump(response, f, indent=2)
    print("Response saved to deepseek_response_object.json")


if __name__ == "__main__":
    main()
