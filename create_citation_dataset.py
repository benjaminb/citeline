import csv
import os
import pysbd
import re
from tqdm import tqdm
from utils import load_dataset

PATH_TO_DATA = 'data/processed_for_chroma/reviews'
# FILENAMES = os.listdir(PATH_TO_DATA)
FILE_PATHS = [os.path.join(PATH_TO_DATA, filename)
              for filename in os.listdir(PATH_TO_DATA)]
SEG = pysbd.Segmenter(language="en", clean=False)

# Regex Patterns
lastname = r"[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ-]*(?:'[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ-]*)?"
year = r"\(?(\d{4}[a-z]?)\)?"
name_sep = r",?\s"
INLINE_CITATION_PATTERN = fr"({lastname}(?:{name_sep}{lastname})*(?: et al.?)?)\s*{year}"

# Compile the regex pattern
inline_regex = re.compile(INLINE_CITATION_PATTERN)


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
    return [match.groups() for match in inline_regex.finditer(text)]


def make_samples_from_record(record: dict, segmenter: pysbd.Segmenter) -> list[dict[str, str]]:
    sentences = segmenter.segment(record['body'])
    samples = []
    for sentence in sentences:
        bibcodes = get_bibcodes_from_inline(sentence, record['reference'])
        if bibcodes:
            samples.append(
                {'doi': record['doi'][0], 'sentence': sentence, 'bibcodes': bibcodes})
    return samples


def main():
    samples = []
    print(FILE_PATHS)
    for filename in FILE_PATHS:
        data = load_dataset(filename)
        for record in tqdm(data[:3]):
            samples += make_samples_from_record(record, SEG)

    # Write out to CSV
    with open('data/samples.json', 'w') as file:
        csv_writer = csv.DictWriter(
            file, fieldnames=['doi', 'sentence', 'bibcodes'])
        csv_writer.writeheader()
        csv_writer.writerows(samples)


if __name__ == "__main__":
    main()
