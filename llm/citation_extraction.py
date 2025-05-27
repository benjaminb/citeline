from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json
from models import SentenceNoCitation, SentenceValidation, CitationList, CitationExtraction
from pydantic import BaseModel

# MODEL_NAME = "llama3.3:latest"  # Replace with your model name
MODEL_NAME = "mistral-nemo:latest"  # Replace with your model name
VALID_SENT_PROMPT = "is_sentence_good_prompt.txt"
CIT_EXTRACT_PROMPT = "citation_prompt.txt"
SENT_NO_CIT_PROMPT = "sent_prompt.txt"


def get_llm_function(
    model_name: str = MODEL_NAME, system_prompt_path: str = None, output_model: BaseModel = None
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


is_sentence_valid = get_llm_function(
    model_name=MODEL_NAME, system_prompt_path=VALID_SENT_PROMPT, output_model=SentenceValidation
)

extract_citations = get_llm_function(
    model_name=MODEL_NAME,
    system_prompt_path=CIT_EXTRACT_PROMPT,
    # output_model=CitationList,
    output_model=CitationExtraction,
)

get_sent_no_citation = get_llm_function(
    model_name=MODEL_NAME,
    system_prompt_path=SENT_NO_CIT_PROMPT,
    output_model=SentenceNoCitation,
)


def main():
    # Example usage
    text = "It also hosts a Compton-thick AGN in the Western component, observed directly in hard X-rays (Della Ceca et al. 2002 ; Ballo et al. 2004 )."
    result = is_sentence_valid(text)
    print(f"Is the sentence valid? {result.is_valid}")
    print(f"Type returned: {type(result.is_valid)}")
    print(f"Reasoning: {result.reasoning}")

    # Extract citations
    citations = extract_citations(text)
    print(f"Extracted citations: {citations.citations}")

    # Get sentence without citations
    sent_no_cit = get_sent_no_citation(text)
    print(f"Sentence without citations: {sent_no_cit.sentence}")


if __name__ == "__main__":
    main()
