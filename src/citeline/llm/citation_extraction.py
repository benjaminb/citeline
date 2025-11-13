import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

try:
    from citeline.llm.models import SentenceValidation, CitationSubstring, CitationList
except ImportError:
    from citeline.llm.models import SentenceValidation, CitationSubstring, CitationList

MODEL_NAME = "llama3.3:latest"  # Replace with your model name
# MODEL_NAME = "mistral-nemo:latest"  # Replace with your model name
VALID_SENT_PROMPT = "src/citeline/llm/prompts/is_sentence_good_prompt.txt"
CIT_SUBSTRING_PROMPT = "src/citeline/llm/prompts/substring_prompt.txt"
CIT_EXTRACT_PROMPT = "src/citeline/llm/prompts/citation_tuples_prompt.txt"

YEAR_PATTERN = r"^(?:\d{4}|^\d{2})"  # Matches a YYYY or YY pattern, even if followed by a letter, but not "in preparation" or similar


def get_llm_function(model_name: str = MODEL_NAME, system_prompt_path: str = None, output_model: BaseModel = None):
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
    output_model=CitationList,
)


def create_sent_no_cit(text: str, citation_substrings: list[str]) -> str:
    """
    Creates a sentence without inline citations by replacing citation substrings with '[REF]'.
    """
    sent_no_cit = text
    for substring in citation_substrings:
        # Replace substring in text with '[REF]' if it exists, if not try to slice the substring from the right
        if substring in sent_no_cit:
            sent_no_cit = sent_no_cit.replace(substring, "[REF]")
            continue
        elif (stripped_substring := substring.strip()) in sent_no_cit:
            sent_no_cit = sent_no_cit.replace(stripped_substring, "[REF]")
            continue
        for i in range(1, 4):
            if (sliced_substring := stripped_substring[:-i]) in sent_no_cit:
                sent_no_cit = sent_no_cit.replace(sliced_substring, "[REF]")
                break
    return sent_no_cit


def sentence_to_citations(text: str) -> tuple[list[tuple[str, str]], str]:
    """
    Converts a sentence to a list of citations.
    This function uses a sequence of LLM calls to
    1) Check if the sentence is a valid scientific sentence (not a caption, mangled OCR, etc.)
    2) Identify the substrings of the sentence that comprise inline citations
    3) Create a list of citation tuples [(author, year), ...] from the substrings

    Returns:
      - None if the sentence is invalid or any errors in processing
      - ([], text) tuple if no citations found
      - ([(author, year), ...], sent_no_cit) tuple if citations found, where sent_no_cit masks inline citations with '[REF]'
    """
    # Check if the sentence is valid
    is_valid_sentence = True
    print("{is_valid=", end="", flush=True)
    try:
        validity_result = is_sentence_valid(text)
        is_valid_sentence = validity_result.is_valid
    except Exception as e:
        print(f"Error checking sentence validity: {e}")

    print(is_valid_sentence, end=", ")
    if not is_valid_sentence:
        print(f"Invalid sentence: {validity_result.reasoning}", flush=True)
        return None

    # Identify citation substrings
    print("citation_substrings=", end="", flush=True)
    citation_substrings = []
    try:
        citation_substrings = get_citation_substrings(text).root
        assert isinstance(
            citation_substrings, list
        ), "Expected a get_citation_substrings(text).root to return a list of citation substrings"
    except Exception as e:
        print(f"Error extracting citation substrings: {e}")
        return None
    print(citation_substrings, end=", ", flush=True)

    # No citations found, return empty list and original sentence
    if not citation_substrings:
        return [], text

    # Use citation substrings to make structured list of citation tuples
    print("citations=", end="", flush=True)
    all_citations = []
    for s in citation_substrings:
        try:
            citation_extraction = extract_citations(s)
            citations = [
                (citation.author, citation.year)
                for citation in citation_extraction.root
                if re.match(
                    YEAR_PATTERN, citation.year
                )  # Ensure year begins with 4 digits, not 'in preparation' or similar
            ]
            all_citations += citations

        except Exception as e:
            print(f"Error extracting citations from substring '{s}': {e}")
            # If there's an error, we can skip this substring
            continue
    print(all_citations, end="}\n", flush=True)
    print(f"sent_original={text}")

    sent_no_cit = create_sent_no_cit(text, citation_substrings)
    print(f"sent_no_cit={sent_no_cit}", flush=True)
    return all_citations, sent_no_cit


def main():
    # Example usage
    text = "It also hosts a Compton-thick AGN in the Western component, observed directly in hard X-rays (Della Ceca et al. 2002 ; Ballo et al. 2004 )."
    # text = "Models to predict ruwe for an arbitrary binary were developed by this work and by Penoyre et al. (2022,"
    text = "Here we review on preliminary work by Bekki et al. ( 2019 ), who simulated fully-compressible convection inside a rotating spherical shell"
    citations, sent_no_cit = sentence_to_citations(text)
    print(f"Extracted citations: {citations}")
    print(f"Sentence without citations: {sent_no_cit}")


if __name__ == "__main__":
    main()
