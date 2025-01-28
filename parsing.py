import pysbd


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
