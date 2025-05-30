import json
from tqdm import tqdm
import re

try:
    from llm.citation_extraction import sentence_to_citations
except ImportError:
    from citation_extraction import sentence_to_citations


def normalize_sentence(text: str) -> str:
    """
    Normalize sentence by removing whitespace, converting to lowercase,
    and standardizing punctuation for comparison.
    """
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[().,;:]", "", text)
    text = text.lower()
    return text


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


def main():
    # Load test data
    with open("data/etc/processed_passages.json", "r") as file:
        data = json.load(file)
        print(f"Loaded {len(data)} records from {file.name}")

    valid_examples = 0
    citation_successes = 0
    sentence_successes = 0
    both_successes = 0
    failures = []

    for record in tqdm(data):
        sentence = record["sentence"]

        # Skip invalid sentences for this test
        is_valid = record["isValid"]
        if not is_valid:
            continue
        valid_examples += 1
        expected_citation_substrings = record["citation_substrings"]
        expected_citations = set((author, year) for author, year in record["citations"])

        print(
            f"Expected: {{is_valid={is_valid}, citation_substrings={expected_citation_substrings}, citations={expected_citations}}}",
            flush=True,
        )
        # Get predictions
        print("Predictd: ", end="", flush=True)
        try:
            predicted_citations, predicted_sent_no_cit, predicted_substrings = sentence_to_citations(sentence)
        except Exception as e:
            print(f"Error processing sentence: {e}")
            with open("failed_processing.jsonl", "a") as f:
                record["error"] = str(e)
                f.write(json.dumps(record) + "\n")
            continue

        # Compare citations
        predicted_citations_set = set(predicted_citations)

        # citation_match = expected_citations == predicted_citations_set
        citation_match = citation_sets_match(predicted_citations_set, expected_citations)
        if citation_match:
            citation_successes += 1
        else:
            print(f"Citation mismatch:")
            print(f"  Sentence: {sentence}")
            print(f"  Expected: {expected_citations}")
            print(f"  Predicted: {predicted_citations_set}")
            record["predicted_citations"] = list(predicted_citations_set)
            failures.append(record)
            with open("failed_citations.jsonl", "a") as f:
                record["predicted_citations"] = list(predicted_citations_set)
                f.write(json.dumps(record) + "\n")

        # Compare sentences without citations
        expected_sent_no_cit = normalize_sentence(record["sent_no_cit"])
        predicted_sent_no_cit_norm = normalize_sentence(predicted_sent_no_cit)

        sentence_match = expected_sent_no_cit == predicted_sent_no_cit_norm
        if sentence_match:
            sentence_successes += 1
        else:
            print(f"Sentence mismatch:")
            print(f"  Original: {sentence}")
            print(f"  Expected: '{expected_sent_no_cit}'")
            print(f"  Predicted: '{predicted_sent_no_cit_norm}'")
            with open("failed_sentences.jsonl", "a") as f:
                record["predicted_sent_no_cit"] = predicted_sent_no_cit
                record["predicted_sent_no_cit_norm"] = predicted_sent_no_cit_norm
                f.write(json.dumps(record) + "\n")

        # Track cases where both succeed
        if citation_match and sentence_match:
            both_successes += 1

        print("}")  # Close the print statement from sentence_to_citations

    # Print results
    with open("llm/failed_examples.json", "w") as f:
        json.dump(failures, f, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"Valid examples: {valid_examples}")
    print(
        f"Citation successes: {citation_successes} ({citation_successes/valid_examples*100:.2f}%)"
    )
    print(
        f"Sentence successes: {sentence_successes} ({sentence_successes/valid_examples*100:.2f}%)"
    )
    print(f"Both successes: {both_successes} ({both_successes/valid_examples*100:.2f}%)")


if __name__ == "__main__":
    main()
