import json
from citation_extraction import get_sent_no_citation
from tqdm import tqdm
import re

with open("../data/etc/processed_passages.json", "r") as file:
    data = json.load(file)
    print(f"Loaded {len(data)} records from {file.name}")


def normalize_sent(text: str) -> str:
    """
    Remove all whitespace, parentheses, commas, and periods
    """
    # s = re.sub(r"\s+", "", text)
    s = re.sub(r"[().,\s+]", "", text)
    return s.lower()


valid_examples = 0
successes = 0

for record in tqdm(data):
    if not record["isValid"]:
        continue

    valid_examples += 1
    sentence = record["sentence"]
    expected = record["sent_no_cit"]
    predicted = get_sent_no_citation(sentence).sentence
    norm_expected, norm_predicted = normalize_sent(expected), normalize_sent(predicted)

    if norm_expected != norm_predicted:
        # Log the failed example
        print(f"Failed example: {record['sentence']}")
        print(f"Expected: {expected}")
        print(f"Predictd: {predicted}")
        print("===========================")
        print(f"Normalized expected: {norm_expected}")
        print(f"Normalized predictd: {norm_predicted}")
        print("===========================")
        with open("failed_citations.jsonl", "a") as f:
            # write out the record
            f.write(json.dumps(record) + "\n")
    else:
        successes += 1

print(f"Valid examples: {valid_examples}")
print(f"Successes: {successes}")
print(f"Success rate: {successes / valid_examples * 100:.2f}%")
