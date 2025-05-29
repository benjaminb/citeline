import json
from tqdm import tqdm
import re

try:
    from llm.citation_extraction import get_citation_substrings
except ImportError:
    from citation_extraction import get_citation_substrings

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


valid_examples = 0
successes = 0

for record in tqdm(data):
    if not record["isValid"]:
        continue

    valid_examples += 1
    sentence = record["sentence"]
    sent_no_cit = record["sent_no_cit"]
    expected = set(record["citation_substrings"])
    predicted = set(get_citation_substrings(sentence).root)

    # Check inline citation recognition
    if expected != predicted:
        # Log the failed example
        print(f"Failed example: {record['sentence']}")
        print(f"Expected: {expected}")
        print(f"Predictd: {predicted}")
        with open("failed_substrings.jsonl", "a") as f:
            # write out the record
            f.write(json.dumps(record) + "\n")
    else:
        successes += 1


print(f"Valid examples: {valid_examples}")
print(f"--- Substring successes ---")
print(f"Successes: {successes}")
print(f"Success rate: {successes / valid_examples * 100:.2f}%")
