"""
This creates an OpenAI client which also can be used for DeepSeek
"""

from openai import OpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from llm.models import IsValidCitation

# Load environment variables from .env file from project root
load_dotenv("../.env")


def deepseek_client():
    assert "DEEPSEEK_API_KEY" in os.environ, "DEEPSEEK_API_KEY must be set in environment variables"
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    return client


with open("llm/prompts/deepseek_citation_identification.txt", "r") as f:
    DEEPSEEK_CITATION_IDENTIFICATION_PROMPT = f.read()

deepseek_llm = ChatDeepSeek(
    model="deepseek-chat", temperature=0.0, max_tokens=10, max_retries=2
).with_structured_output(IsValidCitation, method="json_mode", strict=True)


def deepseek_citation_validator_using_openai(query: str, candidates: list[str]) -> list[bool]:
    """
    Using the OpenAI client, returns a list of booleans indicating whether each candidate is a valid reference.
    """
    client = deepseek_client()
    prompts = [
        DEEPSEEK_CITATION_IDENTIFICATION_PROMPT.format(sentence=query, paper=paper)
        for paper in candidates
    ]
    results = [
        client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.0,
            messages=[{"role": "system", "content": prompt}],
            stream=False,
        )
        for prompt in prompts
    ]
    return results


def deepseek_citation_validator(query: str, candidates: list[str]) -> list[bool]:
    """
    Returns a list of booleans indicating whether each candidate is a valid reference.
    """
    results = []
    prompts = [
        DEEPSEEK_CITATION_IDENTIFICATION_PROMPT.format(sentence=query, paper=paper)
        for paper in candidates
    ]
    messages = [[SystemMessage(content=prompt)] for prompt in prompts]
    try:
        responses = deepseek_llm.batch(messages)
        results = [response.is_valid for response in responses]
    except Exception as e:
        print(f"Error during citation validation: {e}")
        # If there's an error, we assume all candidates are invalid
        results = [False] * len(candidates)
    return results


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
