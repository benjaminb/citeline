import json
from tqdm import tqdm
import re

try:
    from llm.citation_extraction import extract_citations
except ImportError:
    from citation_extraction import extract_citations

with open("data/etc/processed_passages.json", "r") as file:
    data = json.load(file)
    print(f"Loaded {len(data)} records from {file.name}")


def normalize_sent(text: str) -> str:
    """
    Remove all whitespace, parentheses, commas, and periods
    """
    # s = re.sub(r"\s+", "", text)
    s = re.sub(r"[().,\s+]", "", text)
    return s.lower()


def clean_substrings(strings):
    results = set()
    for s in strings:
        if s[0] == "(" and s[-1] == ")":
            s = s[1:-1]
        results.add(s)
    return results


def equivalent_citations(predicted, expected) -> bool:
    """
    Check if the predicted citations match the expected citations.
    """
    # Check that the year strings match
    pred_year, expected_year = predicted[1][:4], expected[1][:4]
    if pred_year != expected_year:
        return False

    # Just check the first four characters of the author name
    pred_author, expected_author = predicted[0].lower(), expected[0].lower()
    if pred_author[:4] != expected_author[:4]:
        return False
    return True


def citation_sets_match(pred_set, exp_set) -> bool:
    """
    Check if the predicted citation set matches the expected citation set.
    """
    if len(pred_set) != len(exp_set):
        return False

    for pred_cit in pred_set:
        if not any(equivalent_citations(pred_cit, exp_cit) for exp_cit in exp_set):
            return False
    return True


valid_examples = 0
successes = 0
failed_examples = []

for record in tqdm(data):
    if not record["isValid"]:
        continue

    valid_examples += 1
    substrings = record["citation_substrings"]
    expected_citations = set((author, year) for author, year in record["citations"])
    predicted_citations = set(
        (citation.author, citation.year) for citation in extract_citations(sentence).root
    )

    # Check inline citation recognition
    if not citation_sets_match(predicted_citations, expected_citations):
        # Log the failed example
        print(f"Failed example: {record['sentence']}")
        print(f"Expected: {expected_citations}")
        print(f"Predictd: {predicted_citations}")
        record["predicted_substrings"] = list(predicted_citations)
        failed_examples.append(record)
    else:
        successes += 1

with open("llm/failed_substrings.json", "w") as f:
    json.dump(failed_examples, f, indent=2)

print(f"Valid examples: {valid_examples}")
print(f"--- Substring successes ---")
print(f"Successes: {successes}")
print(f"Success rate: {successes / valid_examples * 100:.2f}%")
