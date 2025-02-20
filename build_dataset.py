import json
import os
import re
from tqdm import tqdm
from parsing import get_inline_citations, INLINE_CITATION_REGEX
from utils import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

REVIEW_JOURNAL_BIBCODES = {'RvGeo', 'SSRv.', 'LRSP.',
                           'NewAR', 'ESRv.', 'NRvEE', 'P&SS.', 'ARA&A', 'A&ARv'}


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


def sentence_to_example(record, sentence, index, all_records):
    """
    Takes all the inline citations of a sentence and if it can resolve them to dois
    then it returns the 
    """
    # print(f"Working on sentence {index} of {record['doi'][0]}")

    def citation_to_doi(citation):
        """
        Takes a single inline citation as tuple of (author, year) and determines if there is a unique
        matching bibcode in the record's references. If so, it continues to look for a unique
        doi matching that bibcode in the entire dataset. It returns the doi if resolved, otherwise None.
        """
        bibcodes = bibcode_matches(citation, record['reference'])
        if len(bibcodes) != 1:
            return None

        # Take the bibcode and look for a unique corresponding doi AND that the citation isn't to a review journal
        matching_dois = [record['doi'][0]
                         for record in all_records if record['bibcode'] == bibcodes[0]
                         and record['bibcode'][4:9] not in REVIEW_JOURNAL_BIBCODES]
        if len(matching_dois) != 1:
            return None
        return matching_dois[0]

    inline_citations = get_inline_citations(sentence)

    # if not (inline_citations := get_inline_citations(sentence)):
    #     return 0
    citation_dois = []
    for citation in inline_citations:
        if not (doi := citation_to_doi(citation)):
            return None
        citation_dois.append(doi)

    # If all citations resolved to dois, return the example
    # TODO: is this too strict?
    # if len(inline_citations) != len(citation_dois): # this is already accounted for in the loop above?
    #     return -1
    return {
        'source_doi': record['doi'][0],
        'sent_original': sentence,
        'sent_no_cit': re.sub(INLINE_CITATION_REGEX, '', sentence),
        'sent_idx': index,
        'citation_dois': citation_dois
    }


def examples_from_record(record, all_records):
    return [
        example
        for i, sentence in enumerate(record['body_sentences'])
        if (example := sentence_to_example(record, sentence, i, all_records))
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
        records = load_dataset('data/sentence_segmented_and_merged/' + dataset)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(get_examples_fn, record)
                       for record in records]
            for future in tqdm(as_completed(futures), total=len(futures)):
                examples = future.result()

                # Write out examples to jsonl
                for example in examples:
                    # Not all citations resolved
                    if example is None:
                        continue
                    destination = 'data/dataset/no_reviews/nontrivial.jsonl' if len(
                        example['citation_dois']) > 0 else 'data/dataset/no_reviews/trivial.jsonl'
                    with open(destination, 'a') as f:
                        f.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    main()
