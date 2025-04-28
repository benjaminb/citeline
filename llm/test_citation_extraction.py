import json
from citation_extraction import extract_citations
from tqdm import tqdm
from typing import Literal

with open("../data/etc/processed_passages.json", "r") as file:
    data = json.load(file)
    print(f"Loaded {len(data)} records from {file.name}")


def mismatch(predicted, expected) -> Literal["author", "year", "citation_num", None]:
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


valid_examples = 0
successes = 0

for record in tqdm(data):
    if not record["isValid"]:
        continue

    valid_examples += 1
    sentence = record["sentence"]
    expected_citations = record["citations"]
    predicted_citations = extract_citations(sentence)
    mismatches = mismatch(predicted_citations, expected_citations)

    if mismatches:
        # Log the failed example
        print(f"Failed example: {record['sentence']}")
        print(f"Expected: {expected_citations}")
        print(f"Predicted: {predicted_citations}")
        print(f"Mismatches: {mismatches}")
        record["predicted_citations"] = predicted_citations
        record["mismatches"] = mismatches
        with open("failed_citations.jsonl", "a") as f:
            # write out the record
            f.write(json.dumps(record) + "\n")
    else:
        successes += 1

print(f"Valid examples: {valid_examples}")
print(f"Successes: {successes}")
print(f"Success rate: {successes / valid_examples * 100:.2f}%")
