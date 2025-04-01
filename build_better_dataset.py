import json
import os
import pandas as pd
import re
from tqdm import tqdm
from parsing import get_inline_citations, INLINE_CITATION_REGEX

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


# Create a function to build a lookup index
def build_bibcode_index(reference_records):
    """Build an index mapping bibcodes to records for fast lookup"""
    # Pre-filter non-review journals
    filtered_records = {}
    for ref in reference_records:
        bibcode = ref.get("bibcode")
        if bibcode and bibcode[4:9] not in REVIEW_JOURNAL_BIBCODES:
            if bibcode not in filtered_records:
                filtered_records[bibcode] = []
            filtered_records[bibcode].append(ref)
    return filtered_records


# Cache regex patterns
# @lru_cache(maxsize=2048)
def bibcode_regex(author: str, year: str):
    """
    Given first author and year, return a regex pattern for the
    corresponding bibcode. Results are cached for performance.
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


# Modify examples_from_record to accept the index
def examples_from_record_with_index(record, bibcode_index):
    # Use bibcode_index instead of scanning the entire list
    return [
        example
        for i, sentence in enumerate(record["body_sentences"])
        if (example := sentence_to_example_with_index(record, sentence, i, bibcode_index))
    ]


# Modified sentence_to_example function using the index
def sentence_to_example_with_index(record, sentence, index, bibcode_index):
    def citation_to_doi_and_bibcode(citation):
        bibcodes = bibcode_matches(citation, record["reference"])
        if len(bibcodes) != 1:
            return None

        bib = bibcodes[0]

        # Fast O(1) lookup using the index
        if bib not in bibcode_index or len(bibcode_index[bib]) != 1:
            return None
        doi = bibcode_index[bib][0]["doi"]

        if doi:
            return doi, bib
        return None

    # Remove inline citations from the sentence, skip if result is too short (chose 63 after some inspection)
    sent_no_citation = re.sub(INLINE_CITATION_REGEX, "", sentence).strip()
    if len(sent_no_citation) < 63:
        return None

    # Rest of the function remains the same
    inline_citations = get_inline_citations(sentence)
    citation_dois, bibcodes = [], []

    # If ANY inline citation is not found, return None
    for citation in inline_citations:
        result = citation_to_doi_and_bibcode(citation)
        if not result:
            return None
        doi, bib = result
        citation_dois.append(doi)
        bibcodes.append(bib)

    return {
        "source_doi": record["doi"],
        "sent_original": sentence,
        "sent_no_cit": sent_no_citation,
        "sent_idx": index,
        "citation_dois": citation_dois,
        "pubdate": record["pubdate"],
        "resolved_bibcodes": bibcodes,
    }


# Modify the main function to build the index
def main():
    # Load data
    research = pd.read_json("data/preprocessed/research.jsonl", lines=True)
    reviews = pd.read_json("data/preprocessed/reviews.jsonl", lines=True)
    print(f"Loaded {len(research)} research records and {len(reviews)} review records.")

    # Convert DataFrames to lists of dictionaries
    research_dicts = research.to_dict("records")
    reviews_dicts = reviews.to_dict("records")

    # Build the index for fast lookup by bibcode
    print("Building bibcode index for faster lookups...", end="")
    bibcode_index = build_bibcode_index(research_dicts)
    print("done.")

    del research  # Free memory
    del reviews

    trivial_examples, nontrivial_examples = [], []
    for record in tqdm(reviews_dicts, total=len(reviews_dicts), desc="Processing records"):
        examples = examples_from_record_with_index(record, bibcode_index)
        for example in examples:
            if example is None:
                continue
            elif len(example["citation_dois"]) == 0:
                trivial_examples.append(example)
            else:
                nontrivial_examples.append(example)

    # Write results
    print(
        f"Writing {len(trivial_examples)} trivial and {len(nontrivial_examples)} nontrivial examples..."
    )
    os.makedirs("data/dataset", exist_ok=True)
    with open("data/dataset/trivial.jsonl", "w") as f:
        for example in trivial_examples:
            f.write(json.dumps(example) + "\n")
    with open("data/dataset/nontrivial.jsonl", "w") as f:
        for example in nontrivial_examples:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
    # import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()  # Run your main function
    # profiler.disable()

    # stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    # stats.print_stats(40)
