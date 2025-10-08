"""
This creates an OpenAI client which also can be used for DeepSeek
"""

from openai import OpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os
from citeline.llm.models import Findings

# Load environment variables from .env file from project root
load_dotenv("../../../.env")


def github_llm_client(model: str):
    assert "GITHUB_TOKEN" in os.environ, "GITHUB_TOKEN must be set in environment variables"
    print(f"Got GITHUB_TOKEN: {os.environ['GITHUB_TOKEN']}")

    client = OpenAI(api_key=os.environ.get("GITHUB_TOKEN"), base_url="https://models.github.ai/inference")

    def get_llm_response(prompt: str):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            stream=False,
            max_tokens=60000
        )

        return response.choices[0].message.content

    return get_llm_response

def openai_llm_client(model: str):
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY must be set in environment variables"

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_llm_response(prompt: str):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            stream=False,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    return get_llm_response


def main():
    llm = openai_llm_client(model="gpt-5-nano")
    with open("../llm/prompts/original_contributions_v3.txt", "r") as f:
        prompt_template = f.read()

    with open("../../../data/paper_bodies/lacey.txt", "r") as f:
        paper = f.read()
    prompt = prompt_template.format(paper=paper)
    print(f"Prompt:\n{prompt}\n")
    response = llm(prompt=prompt)
    print("Raw response")
    print(response)

    import json
    try:
        response = json.loads(response)
        print(f"Parsed JSON: {response}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")


    with open("deepseek_response_object.json", "w") as f:
        json.dump(response, f, indent=2)
    print("Response saved to deepseek_response_object.json")


if __name__ == "__main__":
    main()
