import datetime
import logging
import re
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import Annotated, TypedDict
from citeline.llm.models import CitationList, CitationSubstring, SentenceUsabilityResponse
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
MODEL_NAME = "gpt-oss:20b"  # Replace with your model name


# --- LOGGING SETUP ---
LOG_DIR = CURRENT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOG_DIR / f"sentence_processor_agent_{timestamp}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- PATTERNS ---
YEAR_PATTERN = r"^(?:\d{4}|^\d{2})"  # Matches a YYYY or YY pattern, even if followed by a letter, but not "in preparation" or similar


class StateDict(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    raw_sentence: str
    is_usable: bool
    usability_reasoning: str
    citation_substring_list: list[str]
    inline_citations: list[tuple[str, int]]
    sent_masked: str
    sent_no_cit: str
    dois: list[str]


def create_llm(model_name: str, system_prompt_path: Path, output_model: type[BaseModel]):
    """
    Creates:
    1. a ChatOllama instance with structured output based on the output_model
    2. Reads in the system prompt from the given path
    3. Binds the system prompt and llm into a function that takes in text as the HumanMessage content, and returns the llm invocation result
    4. Returns this function for repeated use
    """
    llm = ChatOllama(
        model=model_name,
        temperature=0.0,
    ).with_structured_output(output_model, method="json_schema")

    # Read in the system prompt
    with open(system_prompt_path, "r") as f:
        prompt = f.read()
    system_message = SystemMessage(content=prompt)

    # Bind system prompt and llm into a function
    def llm_function(text: str):
        return llm.invoke([system_message, HumanMessage(content=text)])

    return llm_function


sentence_validator_agent = create_llm(
    model_name=MODEL_NAME,
    system_prompt_path=(CURRENT_DIR / ".." / "llm" / "prompts" / "is_sentence_valid_prompt.txt").resolve(),
    output_model=SentenceUsabilityResponse,
)

# Sentence -> list[str], the substrings of inline citations as they appear
citation_substring_agent = create_llm(
    model_name=MODEL_NAME,
    system_prompt_path=(CURRENT_DIR / ".." / "llm" / "prompts" / "substring_prompt.txt").resolve(),
    output_model=CitationSubstring,
)

# Inline citation substrings -> [(Author, Year), ...] list of tuples
citation_tuples_agent = create_llm(
    model_name=MODEL_NAME,
    system_prompt_path=(CURRENT_DIR / ".." / "llm" / "prompts" / "citation_tuples_prompt.txt").resolve(),
    output_model=CitationList,
)


def check_sentence_usability(state: StateDict) -> StateDict:
    """
    Checks if the sentence is usable for the dataset. See its system prompt for full details. In short:
    - Not a caption, OCR gibberish, or end matter / reference section
    - Expresses a scientific thought
    """

    # load the system and human prompt templates
    sentence = state["raw_sentence"]
    try:
        usability_response = sentence_validator_agent(sentence)
    except Exception as e:
        logger.error(f"Error validating sentence: {e}")
        return state  # Return state unchanged on error

    # Add results to state
    state["usability_reasoning"] = usability_response.reasoning
    state["is_usable"] = usability_response.is_usable

    return state


def route_after_sent_validation(state: StateDict) -> str:
    if state.get("is_usable", False):
        return "write_citation_substrings"
    return END


def write_citation_substrings(state: StateDict) -> StateDict:
    sentence = state["raw_sentence"]
    try:
        citation_substrings_response = citation_substring_agent(sentence)
    except Exception as e:
        logger.error(f"Error extracting citation substrings: {e}")
        return state  # Return state unchanged on error

    state["citation_substring_list"] = [c.strip() for c in citation_substrings_response.root]

    return state


def route_after_substrings_written(state: StateDict) -> str:
    if not state.get("citation_substring_list"):
        return END
    return "substrings_to_tuples"


def substrings_to_tuples(state: StateDict) -> StateDict:
    substrings = state["citation_substring_list"]
    try:
        all_citation_tuples = []
        # Each substring could be 1 or more inline citations
        for substring in substrings:
            citation_tuples_response = citation_tuples_agent(substring)
            citation_tuples = [
                (cit.author, cit.year) for cit in citation_tuples_response.root if re.match(YEAR_PATTERN, cit.year)
            ]
            all_citation_tuples += citation_tuples
    except Exception as e:
        logger.error(f"Error converting substrings to citation tuples: {e}")
        return state  # Return state unchanged on error

    state["inline_citations"] = all_citation_tuples
    return state


def mask_and_strip_sentence(state: StateDict) -> StateDict:
    raw_sentence = state["raw_sentence"]
    substrings = state["citation_substring_list"]

    for substring in substrings:
        # Try for the biggest possible match first
        if substring in raw_sentence:
            sent_masked = raw_sentence.replace(substring, "[REF]")
            continue
        # Try stripping whitespace around substring and replacing that
        elif (stripped_substring := substring.strip()) in raw_sentence:
            sent_masked = raw_sentence.replace(stripped_substring, "[REF]")
            continue

        # Try truncating up to 3 chars from the end (punctuation inconsistencies, etc.)
        for i in range(1, 4):
            if (sliced_substring := stripped_substring[:-i]) in raw_sentence:
                sent_masked = raw_sentence.replace(sliced_substring, "[REF]")
                break  # Stop after first successful match, which will be the largest possible match

    # Create sent_no_cite by removing [REF] tokens and cleaning possible leftover parens
    sent_no_cit = sent_masked.replace("[REF]", "").strip()
    sent_no_cit = re.sub(r"\s*\(\s*\)", "", sent_no_cit)  # Consumes empty parens and left whitespace

    state["sent_masked"] = sent_masked
    state["sent_no_cit"] = sent_no_cit  # In this case, masked and no_cit are the same

    return state


# --- THE GRAPH ---
builder = StateGraph(StateDict)

builder.add_node(check_sentence_usability)
builder.add_node(write_citation_substrings)
builder.add_node(substrings_to_tuples)
builder.add_node(mask_and_strip_sentence)

builder.add_edge(START, "check_sentence_usability")

# if not usable, go to END, otherwise go to write_citation_substrings
builder.add_conditional_edges("check_sentence_usability", route_after_sent_validation)

# If citation substrings is an empty list go to END (you have an input sent with no citations)
builder.add_conditional_edges("write_citation_substrings", route_after_substrings_written)

builder.add_edge("substrings_to_tuples", "mask_and_strip_sentence")
builder.add_edge("mask_and_strip_sentence", END)

sentence_processor_agent = builder.compile()


def main():
    input_state = {
        "raw_sentence": "We also give the ‘natural volume’ Vf that is associated with a filter of radius Ru defined to be the integral of WR( r)[ W^(0) over all space."
    }
    final_state = sentence_processor_agent.invoke(input_state)
    from pprint import pprint

    pprint(final_state)


if __name__ == "__main__":
    main()
