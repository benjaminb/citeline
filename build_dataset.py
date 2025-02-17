import json
import os
import re
from tqdm import tqdm
from parsing import get_inline_citations
from utils import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


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
    Given an inline citation and a list of references, return the references
    h the inline citation's bibcode regex pattern
    """
    pattern = bibcode_regex(*inline_citation)
    return [s for s in references if pattern.match(s)]


def sentence_to_example(record, sentence, all_records):
    """
    Takes all the inline citations of a sentence and if it can resolve them to dois
    then it returns the """
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


def examples_from_record(record, all_records):
    return [
        example for sentence in record['body_sentences'] if (example := sentence_to_example(record, sentence, all_records))
    ]


def get_all_records(paths: list[str]) -> list[dict]:
    """
    `all_records` used in examples_from_record reall only needs the bibcode and doi keys, so
    here we filter out all other keys to keep memory demand low
    """
    return [
        {'doi': record['doi'], 'bibcode': record['bibcode']}
        for path in paths
        for record in load_dataset('data/json/' + path)
    ]


def main():
    all_records = get_all_records(
        [f for f in os.listdir('data/json/') if f.endswith('.json')])

    # Bind all_records to examples_from_record, since executor.submit doesn't allow for multiple args
    get_examples_fn = partial(examples_from_record, all_records=all_records)

    for dataset in ['Astro_Reviews.json', 'Earth_Science_Reviews.json', 'Planetary_Reviews.json']:
        print(f"Processing {dataset}...")
        records = load_dataset('data/processed/' + dataset)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(get_examples_fn, record)
                       for record in records]
            for future in tqdm(as_completed(futures), total=len(futures)):
                examples = future.result()

                # Write out examples to jsonl
                for example in examples:
                    destination = 'data/dataset/nontrivial.jsonl' if len(
                        example['citation_dois']) > 0 else 'data/dataset/trivial.jsonl'
                    with open(destination, 'a') as f:
                        f.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    main()
