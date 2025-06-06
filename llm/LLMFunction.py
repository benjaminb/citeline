from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Union
from pydantic import BaseModel, RootModel

MODEL_NAME = "mistral-nemo:latest"  # Replace with your model name


class LLMFunction:
    def __init__(self, model_name, system_prompt_path, output_model):
        """
        Initialize the LLMFunction with the model name, system prompt path, and output model.
        """
        self.llm_function = get_llm_function(
            model_name=model_name,
            system_prompt_path=system_prompt_path,
            output_model=output_model,
        )

    def __call__(self, text: str):
        """
        Call the LLM function with the provided text.
        :param text: The input text to process.
        :return: The output from the LLM function.
        """
        return self.llm_function(text)


def get_llm_function(
    model_name: str = MODEL_NAME,
    system_prompt_path: str = None,
    output_model: Union[BaseModel, RootModel] = None,
):
    """
    Returns a function that invokes the LLM with the given model name and system prompt.
    """
    llm = ChatOllama(
        model=model_name,
        temperature=0.0,
    ).with_structured_output(output_model, method="json_schema")

    # Read in the system prompt
    with open(system_prompt_path, "r") as f:
        sys_msg = f.read()
    sys_msg = SystemMessage(content=sys_msg)

    def llm_function(text: str):
        try:
            msg = HumanMessage(content=text)
            return llm.invoke([sys_msg, msg])
        except Exception as e:
            print(f"Exception: {e}")
            return None

    return llm_function
