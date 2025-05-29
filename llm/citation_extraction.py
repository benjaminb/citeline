import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

try:
    from llm.models import SentenceValidation, CitationExtraction, CitationSubstring
except ImportError:
    from models import SentenceValidation, CitationExtraction, CitationSubstring

MODEL_NAME = "llama3.3:latest"  # Replace with your model name
# MODEL_NAME = "mistral-nemo:latest"  # Replace with your model name
VALID_SENT_PROMPT = "llm/prompts/is_sentence_good_prompt.txt"
CIT_SUBSTRING_PROMPT = "llm/prompts/substring_prompt.txt"
CIT_EXTRACT_PROMPT = "llm/prompts/citation_prompt.txt"
SENT_NO_CIT_PROMPT = "llm/prompts/sent_prompt.txt"

YEAR_PATTERN = r"^\d{4}"  # Matches a year that starts with 4 digits, e.g., "2023a", "1999", but not "in preparation" or similar

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

get_citation_substrings = get_llm_function(
    model_name=MODEL_NAME,
    system_prompt_path=CIT_SUBSTRING_PROMPT,
    output_model=CitationSubstring,
)

extract_citations = get_llm_function(
    model_name=MODEL_NAME,
    system_prompt_path=CIT_EXTRACT_PROMPT,
    output_model=CitationExtraction,
)


def sentence_to_citations(text: str):
    """
    Converts a sentence to a list of citations.
    This function uses a sequence of LLM calls to
    1) Check if the sentence is a valid scientific sentence (not a caption, mangled OCR, etc.)
    2) Identify the substrings of the sentence that comprise inline citations
    3) Create a list of citation tuples [(author, year), ...] from the substrings
    """
    # Check if the sentence is valid
    is_valid_sentence = True
    try:
        validity_result = is_sentence_valid(text)
        is_valid_sentence = validity_result.is_valid
    except Exception as e:
        print(f"Error checking sentence validity: {e}")

    if not is_valid_sentence:
        return []

    # If the sentence is valid, identify citation substrings
    try:
        citation_substrings = get_citation_substrings(text).root
        assert isinstance(
            citation_substrings, list
        ), "Expected a get_citation_substrings(text).root to return a list of citation substrings"
    except Exception as e:
        print(f"Error extracting citation substrings: {e}")
        return []

    # Use citation substrings to make structured list of citation tuples
    all_citations = []
    for s in citation_substrings:
        try:
            citation_extraction = extract_citations(s)
            citations = [
                (citation.author, citation.year) for citation in citation_extraction.citations.root
                if re.match(YEAR_PATTERN, citation.year)  # Ensure year begins with 4 digits, not 'in preparation' or similar
            ]
            all_citations += citations
        except Exception as e:
            print(f"Error extracting citations from substring '{s}': {e}")
            # If there's an error, we can skip this substring
            continue

    # If there are citations, replace their substrings in the original sentence with '[REF]'
    sent_no_cit = citation_extraction.sentence

    return all_citations, sent_no_cit


def main():
    # Example usage
    text = "It also hosts a Compton-thick AGN in the Western component, observed directly in hard X-rays (Della Ceca et al. 2002 ; Ballo et al. 2004 )."

    citations = sentence_to_citations(text)
    print(f"Extracted citations: {citations}")
    # result = is_sentence_valid(text)
    # print(f"Is the sentence valid? {result.is_valid}")
    # print(f"Type returned: {type(result.is_valid)}")
    # print(f"Reasoning: {result.reasoning}")

    # # Extract citations
    # citations = extract_citations(text)
    # print(f"Extracted citations: {citations.citations}")


if __name__ == "__main__":
    main()
