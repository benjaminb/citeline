import json
from tqdm import tqdm
from typing import Literal
import re

try:
    from llm.citation_extraction import (
        is_sentence_valid,
        get_citation_substrings,
        extract_citations,
    )
except ImportError:
    from citation_extraction import is_sentence_valid, get_citation_substrings, extract_citations

with open("data/etc/processed_passages.json", "r") as file:
    data = json.load(file)
    print(f"Loaded {len(data)} records from {file.name}")


def normalize_sent(text: str) -> str:
    """
    Remove all whitespace, parentheses, commas, and periods
    """
    s = re.sub(r"\s+", "", text)
    s = re.sub(r"[().,\s+]", "", text)
    return s.lower()


def equivalent_citations(predicted, expected) -> bool:
    """
    Check if the predicted citations match the expected citations.
    """
    # Check that the year strings match
    pred_year, expected_year = predicted[1][:4], expected[1][:4]
    if pred_year != expected_year:
        print(f"Year mismatch: predicted {pred_year} != expected {expected_year}")
        return False

    # Just check the first four characters of the author name
    pred_author, expected_author = predicted[0].lower(), expected[0].lower()
    if pred_author[:4] != expected_author[:4]:
        print(f"Author mismatch: predicted {pred_author} != expected {expected_author}")
        return False
    return True


valid_examples = 0
successes = 0

for record in tqdm(data):
    sentence = record["sentence"]
    if not record["isValid"]:
        continue

    citation_substrings = record["citation_substrings"]
    expected_citations = set((author, year) for author, year in record["citations"])
    predicted_citations = set(
        (citation.author, citation.year) for citation in extract_citations(sentence).root
    )

    all_predictions_match = True
    for pred_cit in predicted_citations:
        if not any(equivalent_citations(pred_cit, exp_cit) for exp_cit in expected_citations):
            all_predictions_match = False
            print(f"Predicted citation {pred_cit} does not match any expected citation.")
            print(f"Expected citations: {expected_citations}")
            record["expected_citations"] = list(expected_citations)
            record["predicted_citations"] = list(predicted_citations)
            with open("failed_citations.jsonl", "a") as f:
                # write out the record
                f.write(json.dumps(record) + "\n")
            continue
        else:
            print(
                f"Predicted citation {pred_cit} matches an expected citation in {expected_citations}."
            )

    # Get predicted citation substrings -> citation list
    valid_examples += 1
    successes += int(all_predictions_match)

print(f"Valid examples: {valid_examples}")
print(f"--- Substring successes ---")
print(f"Successes: {successes}")
print(f"Success rate: {successes / valid_examples * 100:.2f}%")


def citation_mismatch(predicted, expected) -> Literal["author", "year", "citation_num", None]:
    """
    Check if the predicted citations match the expected citations.
    """
    mismatch_types = []
    if len(predicted) != len(expected):
        mismatch_types.append("citation_num")

    for pred, exp in zip(predicted, expected):
        # Check if first element (name) first letter matches and
        if pred[0][0].upper() != exp[0][0].upper():
            mismatch_types.append("author")
        if pred[1][:4] != exp[1][:4]:
            mismatch_types.append("year")
    return mismatch_types
