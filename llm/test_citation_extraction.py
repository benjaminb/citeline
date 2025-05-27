import json
from citation_extraction import extract_citations
from tqdm import tqdm
from typing import Literal
import re

with open("../data/etc/processed_passages.json", "r") as file:
    data = json.load(file)
    print(f"Loaded {len(data)} records from {file.name}")


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


def normalize_sent(text: str) -> str:
    """
    Remove all whitespace, parentheses, commas, and periods
    """
    # s = re.sub(r"\s+", "", text)
    s = re.sub(r"[().,\s+]", "", text)
    return s.lower()


valid_examples = 0
citation_successes = 0
sentence_successes = 0

for record in tqdm(data):
    if not record["isValid"]:
        continue

    valid_examples += 1
    sentence = record["sentence"]
    sent_no_cit = record["sent_no_cit"]
    expected_citations = record["citations"]
    predicted = extract_citations(sentence)
    pred_citations = [
        [citation.author, citation.year] for citation in predicted.citation_list.citations
    ]
    pred_sentence = predicted.sentence

    # Check inline citation recognition
    mismatches = citation_mismatch(pred_citations, expected_citations)
    if mismatches:
        # Log the failed example
        print(f"Failed example: {record['sentence']}")
        print(f"Expected: {expected_citations}")
        print(f"Predicted: {pred_citations}")
        print(f"Mismatches: {mismatches}")
        record["predicted_citations"] = pred_citations
        record["mismatches"] = mismatches
        with open("failed_citations.jsonl", "a") as f:
            # write out the record
            f.write(json.dumps(record) + "\n")
    else:
        citation_successes += 1

    expected_sent_norm = normalize_sent(sent_no_cit)
    predicted_sent_norm = normalize_sent(pred_sentence)
    if expected_sent_norm != predicted_sent_norm:
        # Log the failed example
        print(f"Failed sentence: {record['sentence']}")
        print(f"Citations identified: {pred_citations}")
        print(f"Expected: {sent_no_cit}")
        print(f"Predicted: {pred_sentence}")
        record["predicted_sentence"] = pred_sentence
        with open("failed_sentences.jsonl", "a") as f:
            # write out the record
            f.write(json.dumps(record) + "\n")
    else:
        sentence_successes += 1


print(f"Valid examples: {valid_examples}")
print(f"--- Citation successes ---")
print(f"Successes: {citation_successes}")
print(f"Success rate: {citation_successes / valid_examples * 100:.2f}%")
print(f"--- Sentence successes ---")
print(f"Successes: {sentence_successes}")
print(f"Success rate: {sentence_successes / valid_examples * 100:.2f}%")
