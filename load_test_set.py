import json


def main():
    with open('data/test_set.jsonl', 'r') as f:
        test_set = [json.loads(line) for line in f.readlines()]

    print(f"Number of test samples: {len(test_set)}")

    samples_with_citations = [sample for sample in test_set if len(
        sample['citation_dois']) > 0]
    print(
        f"Number of samples with 1 or more citation_dois: {len(samples_with_citations)}")
    print(samples_with_citations[:5])


if __name__ == "__main__":
    main()
