import re
from parsing import get_inline_citations


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
