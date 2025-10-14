from datetime import datetime

from openai import OpenAI
import json
import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv("../../../.env")


def bind_client(func):
    """
    Decorator to bind OpenAI client to a function that will provide DeepSeek API access
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    def wrapper(*args, **kwargs):
        return func(client, *args, **kwargs)

    return wrapper


@bind_client
def deepseek_formatted(client, prompt: str, model: str="deepseek-chat") -> str:
    """
    Sends a prompt to the DeepSeek API (using DeepSeek-V3.1 non-thinking model)

    Expects a prompt that will instruct the model to respond with a JSON object.
    However, the function returns the raw string response, to allow for validation and
    error handling in multiple passes without losing the original response
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        stream=False,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def get_deepseek_formatted_response(prompt: str, parse_model: BaseModel, llm_model: str="deepseek-chat") -> BaseModel | None:
    """
    Sends a prompt to the DeepSeek API and attempts to parse the response into the provided Pydantic model.
    If parsing fails, logs the error and returns None.

    Args:
        prompt: The prompt to send to the DeepSeek API
        model: The Pydantic model class to parse the response into
        error_log_path: Path to the log file for recording parsing errors
    Returns:
        An instance of the provided Pydantic model if parsing is successful, otherwise None
    """
    error_log_path = f"deepseek_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Get raw response
    try:
        response = deepseek_formatted(prompt)
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"API call failed: {e}\n")
        return None

    # Attempt to parse JSON
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        with open(error_log_path, "a") as f:
            f.write(f"JSON decode error: {e}\nResponse: {response}\n")
        return None

    return data


def main():
    from citeline.llm.models import Findings

    response = get_deepseek_formatted_response(
        """
Write out the main findings of Attention is All You Need. Output ONLY a single JSON object: {"findings":[ ... ]}. No extra text, no commentary, no additional keys.
        """,
        Findings,
    )
    print(response)


if __name__ == "__main__":
    main()
