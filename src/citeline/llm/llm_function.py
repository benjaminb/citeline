from langchain_ollama import ChatOllama
from typing import Union
from pydantic import BaseModel, RootModel
from citeline.llm.models import LLMResponse

MODEL_NAME = "mistral-nemo:latest"  # Replace with your model name


class LLMFunction:
    """
    Takes a single prompt template and output model, returns a function that can be called with the prompt template's
    variables specified in a dict to get the LLM response

    NOTE: you must provide a dict to the call method based on what variables are required by the prompt template
    and you extract fields as needed based on the output_model
    """

    def __init__(self, model_name: str, prompt_path: str, output_model: Union[BaseModel, RootModel]):
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.output_model = output_model
        with open(prompt_path, "r") as f:
            self.prompt = f.read()
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            # )
        ).with_structured_output(output_model, method="json_schema")

    def __call__(self, prompt_kwargs: dict) -> Union[BaseModel, RootModel]:
        formatted_prompt = self.prompt.format(**prompt_kwargs)
        response = self.llm.invoke(formatted_prompt)
        return response


def main():
    from citeline.llm.models import Findings
    from time import time

    start = time()
    contribution_extractor = LLMFunction(
        model_name="deepseek-r1:70b", prompt_path="prompts/original_contributions_revised.txt", output_model=Findings
    )
    print(f"Initialized contribution extractor in {time() - start:.4f} seconds")
    with open("temp_paper.txt", "r") as f:
        paper = f.read()
    print(f"Loaded paper with {len(paper)} characters", flush=True)
    start = time()
    response = contribution_extractor({"paper": paper})
    print(f"LLM function took {time() - start:.4f} seconds")
    print(f"Got response with {len(response.findings)} findings:")
    for finding in response.findings:
        print(f"- {finding}")


if __name__ == "__main__":
    main()
