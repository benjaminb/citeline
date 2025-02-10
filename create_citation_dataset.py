import csv
import json
import os
import pysbd
import re
from tqdm import tqdm
from utils import load_dataset

PATH_TO_DATA = 'data/json/'
FILE_PATHS = ['data/json/Astro_Reviews.json']
SEG = pysbd.Segmenter(language="en", clean=False)

# Regex Patterns
lastname = r"[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ-]*(?:'[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ-]*)?"
year = r"\(?(\d{4}[a-z]?)\)?"
name_sep = r",?\s"
INLINE_REGEX = re.compile(
    fr"({lastname}(?:{name_sep}{lastname})*(?: et al.?)?)\s*{year}")


def bibcode_regex(author: str, year: str):
    """
    Given first author and year, return a regex pattern for the
    corresponding bibcode
    """
    initial = author[0]
    year = year[:4]  # cut off any letters at the end
    pattern = fr'^{year}.*{initial}$'
    return re.compile(pattern)


def bibcode_matches(inline_citation: tuple[str, str], references: list[str]) -> int:
    """
    Given an inline citation and a list of references, return the number of
    references that match the inline citation's bibcode regex pattern
    """
    pattern = bibcode_regex(*inline_citation)
    return [s for s in references if pattern.match(s)]


def get_bibcodes_from_inline(text: str, references: list[str]) -> list[str]:
    reference_matches = [matches[0] for citation in get_inline_citations(
        text) if len((matches := bibcode_matches(citation, references))) == 1]
    return reference_matches


def get_inline_citations(text: str) -> list[tuple[str, str]]:
    return [match.groups() for match in INLINE_REGEX.finditer(text)]


def sentence_to_example(record, sentence, all_records):
    """
    """
    def citation_to_doi(citation):
        """
        Takes a single inline citation as tuple of (author, year) and determines if there is a unique
        matching bibcode in the record's references. If so, it continues to look for a unique
        doi matching that bibcode in the entire dataset. It returns the doi if resolved, otherwise None.
        """
        bibcodes = bibcode_matches(citation, record['reference'])
        if len(bibcodes) != 1:
            return None

        # Take the bibcode and look for a unique corresponding doi
        matching_dois = [record['doi'][0]
                         for record in all_records if record['bibcode'] == bibcodes[0]]
        if len(matching_dois) != 1:
            return None
        return matching_dois[0]

    inline_citations = get_inline_citations(sentence)
    citation_dois = []
    for citation in inline_citations:
        if not (doi := citation_to_doi(citation)):
            break
        citation_dois.append(doi)

    # If all citations resolved to dois, return the example
    # TODO: is this too strict?
    if len(inline_citations) != len(citation_dois):
        return None
    return {
        'source_doi': record['doi'][0],
        'sentence': sentence,
        'citation_dois': citation_dois
    }


def create_examples_from_record(record, all_records):
    # TODO: come up with a better filter for sentences than just length?
    sentences = [s for s in tqdm(SEG.segment(record['body']), desc=f'Segmenting {record["title"]}...', leave=False)
                 if len(s) > 40]
    return [
        example for sentence in tqdm(sentences, desc='Creating examples from sentences...', leave=False)
        if (example := sentence_to_example(record, sentence, all_records))
    ]


def main():
    # Construct a list of all records doi and bibcode to resolve citations
    all_records = []
    print(f"Constructing test examples from {FILE_PATHS}")
    for filename in FILE_PATHS:
        data = load_dataset(filename)
        all_records += [{'doi': record['doi'],
                         'bibcode': record['bibcode']} for record in data]

    print(f"Size of corpus: {len(all_records)}")

    # Write out to jsonl
    with open('data/test_set.jsonl', 'a') as file:
        for filename in FILE_PATHS:
            data = load_dataset(filename)
            for record in tqdm(data, desc=f"Processing {filename}", leave=False):
                examples = create_examples_from_record(record, all_records)
                for example in examples:
                    json.dump(example, file)
                    file.write('\n')


if __name__ == "__main__":
    main()
