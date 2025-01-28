import json
import os
import re

PATH_TO_DATA = "data/processed_for_chroma"


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


def make_citation_bibcode_list(inline_citations: list[tuple[str, str]], references: list[str]) -> list[tuple[tuple[str, str], str]]:
    """
    Given a paper's list of inline citations and list of references, return a list of
    tuples where the first element is the inline citation and the second element
    is the corresponding bibcode from the references list where there is exactly one match
    """
    return [(citation, matches[0]) for citation in inline_citations
            if len((matches := bibcode_matches(citation, references))) == 1]


def get_json_docs(path: list[str]) -> dict:
    """
    Given a file path, return the json data as a list of dictionaries,
    and deserialize the 'reference' field
    """
    with open(path, 'r') as file:
        data = json.load(file)
    metadatas = data['metadatas']
    return [{**metadatas, 'reference': json.loads(metadatas['reference'])} for metadatas in metadatas]


def get_all_bibcodes(dirs: list[str]) -> list[str]:
    all_records = []
    for directory in dirs:
        filenames = os.listdir(f'{PATH_TO_DATA}/{directory}/')
        print(f"Loading records from {filenames}")
        for filename in filenames:
            records = get_json_docs(os.path.join(
                PATH_TO_DATA, directory, filename))
            all_records.extend(records)
    bibcodes = [record['bibcode'] for record in all_records]
    print(f"Found {len(bibcodes)} bibcodes")
    return bibcodes


def write_all_bibcodes_to_file():
    bibcodes = get_all_bibcodes(['research', 'reviews'])
    with open('data/bibcodes.json', 'w') as file:
        json.dump(bibcodes, file)


def main():
    write_all_bibcodes_to_file()


if __name__ == "__main__":
    main()
