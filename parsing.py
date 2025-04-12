import pysbd
import re

# Regex for inline citations

lastname = r"[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ-]*(?:'[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ-]*)?"
year = r"\(?\s?(\d{4}[a-z]?)\s?\)?"
name_sep = r",?\s| and | & "
INLINE_CITATION_REGEX = re.compile(
    fr"({lastname}(?:{name_sep}{lastname})*(?: et al.?\s?)?),?\s*{year}")


def get_inline_citations(text: str) -> list[tuple[str, str]]:
    matches = [match.groups()
               for match in INLINE_CITATION_REGEX.finditer(text)]
    return [(author, year.strip()) for author, year in matches]


def segment_sentences(text: str) -> list[str]:
    """
    Takes a string input and returns the individual sentences as a list of strings
    """
    return pysbd.Segmenter(language="en", clean=False).segment(text)


def record_body_to_sentences(record: dict) -> list[str]:
    """
    Given a record, parse the body text into individual sentences
    """
    return segment_sentences(record['body'])
