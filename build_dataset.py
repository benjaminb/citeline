import json
import os
import pandas as pd
import re
from tqdm import tqdm
from parsing import get_inline_citations, INLINE_CITATION_REGEX
from concurrent.futures import ProcessPoolExecutor, as_completed

REVIEW_JOURNAL_BIBCODES = {
    "RvGeo",
    "SSRv.",
    "LRSP.",
    "NewAR",
    "ESRv.",
    "NRvEE",
    "P&SS.",
    "ARA&A",
    "A&ARv",
}


def bibcode_regex(author: str, year: str):
    """
    Given first author and year, return a regex pattern for the
    corresponding bibcode
    """
    initial = author[0]
    year = year[:4]  # cut off any letters at the end
    pattern = rf"^{year}.*{initial}$"
    return re.compile(pattern)


def bibcode_matches(inline_citation: tuple[str, str], references: list[str]) -> int:
    """
    Given an inline citation and a list of references, return the references
    h the inline citation's bibcode regex pattern
    """
    pattern = bibcode_regex(*inline_citation)
    return [s for s in references if pattern.match(s)]


def sentence_to_example(
    record: dict, sentence: str, index: int, reference_records: list[dict]
) -> dict | None:
    """
    Takes all inline citations of a sentence, resolves them to DOIs if possible, and returns
    a dictionary representing the example.

    Args:
        record (dict): A dictionary containing keys like "doi", "reference", and "bibcode".
        sentence (str): The sentence to process.
        index (int): The index of the sentence in the record.
        reference_records (list[dict]): A list of dictionaries with keys including "doi" and "bibcode",
                                    used to look up unique DOIs.

    Returns:
        dict | None: A dictionary containing the example data if successful, otherwise None.
    """

    def citation_to_doi(citation: tuple[str, str]) -> str | None:
        """Resolve a single inline citation (author, year) to a DOI if it's uniquely matched."""

        # Attempt to uniquely match an inline citation to a bibcode in the record's references list
        bibcodes = bibcode_matches(citation, record["reference"])
        if len(bibcodes) != 1:
            return None

        bib = bibcodes[0]

        # Find matching references by bibcode
        filtered = [
            ref
            for ref in reference_records
            if ref["bibcode"] == bib and ref["bibcode"][4:9] not in REVIEW_JOURNAL_BIBCODES
        ]

        if len(filtered) != 1:
            return None

        # Extract the DOI from the matched reference
        doi_list = filtered[0]["doi"]
        if not doi_list:
            return None

        return doi_list[0]

    inline_citations = get_inline_citations(sentence)
    citation_dois = []

    for citation in inline_citations:
        # If any citation cannot be resolved, skip this sentence
        doi = citation_to_doi(citation)
        if not doi:
            return None
        citation_dois.append(doi)

    return {
        "source_doi": record["doi"][0],
        "sent_original": sentence,
        "sent_no_cit": re.sub(INLINE_CITATION_REGEX, "", sentence),
        "sent_idx": index,
        "citation_dois": citation_dois,
    }


def examples_from_record(record, reference_records):
    return [
        example
        for i, sentence in enumerate(record["body_sentences"])
        if (example := sentence_to_example(record, sentence, i, reference_records))
    ]


def main():
    # Load data
    research = pd.read_json("data/preprocessed/research.jsonl", lines=True)
    reviews = pd.read_json("data/preprocessed/reviews.jsonl", lines=True)
    print(f"Loaded {len(research)} research records and {len(reviews)} review records.")

    # Convert DataFrames to lists of dictionaries for pickling
    research_dicts = research.to_dict("records")
    reviews_dicts = reviews.to_dict("records")

    # Free up memory from pandas DataFrames
    del research
    del reviews

    # Verify columns in dictionaries
    required_cols = {"body_sentences", "reference", "doi", "bibcode"}
    for name, records in [("research", research_dicts[:5]), ("reviews", reviews_dicts[:5])]:
        for record in records:
            missing = required_cols - set(record.keys())
            if missing:
                print(f"Warning: {name} records are missing columns {missing}")
                break

    trivial_examples, nontrivial_examples = [], []
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        futures = [
            executor.submit(examples_from_record, record, research_dicts)
            for record in reviews_dicts
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing records"):
            # Process results from each worker
            examples = future.result()
            for example in examples:
                if example is None:
                    continue
                elif len(example["citation_dois"]) == 0:
                    trivial_examples.append(example)
                else:
                    nontrivial_examples.append(example)

    print(f"Writing out jsonl...")
    with open("data/dataset/trivial.jsonl", "w") as f:
        for example in trivial_examples:
            f.write(json.dumps(example) + "\n")
    with open("data/dataset/nontrivial.jsonl", "w") as f:
        for example in nontrivial_examples:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
