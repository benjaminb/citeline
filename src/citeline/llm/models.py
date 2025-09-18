from pydantic import BaseModel, RootModel, Field


class SentenceValidation(BaseModel):
    reasoning: str = Field(
        description="Brief reasoning for why the sentence is valid or not. Consider if it's natural language, has clear meaning, and is not a figure caption, table content, reference list item, or gibberish."
    )
    is_valid: bool = Field(
        description="True if the sentence is a 'good' scientific sentence (natural language, clear meaning), False otherwise (e.g., caption, reference, gibberish, OCR error)."
    )


class CitationSubstring(RootModel[list[str]]):
    """
    A substring of a sentence that contains an inline citation.
    This is used to identify the part of the sentence that corresponds to a citation.
    """

    root: list[str] = Field(description="A substring of a sentence that contains an inline citation")


class Citation(BaseModel):
    author: str = Field(description="First author of the cited work")
    year: str = Field(description="Publication year of the cited work, possibly with letters (e.g., '2023a')")


class CitationList(RootModel[list[Citation]]):
    """
    A list of inline citations extracted from the sentence.
    """

    root: list[Citation] = Field(description="List of inline citations extracted from the sentence")


class SentenceNoCitation(BaseModel):
    citations: CitationList = Field(description="List of inline citations extracted from the sentence, if any")
    sentence: str = Field(description="A sentence with any inline citations replaced by '[REF]' placeholders")


class CitationExtraction(BaseModel):
    citations: CitationList = Field(description="List of inline citations extracted from the sentence, if any")
    sentence: str = Field(description="A sentence with any inline citations replaced by '[REF]' placeholders")


class IsValidReference(RootModel[bool]):
    """
    A model to infer if a sentence is a valid scientific sentence.
    """

    root: bool = Field(
        description="True if the sentence is a 'good' scientific sentence (natural language, clear meaning), False otherwise (e.g., caption, reference, gibberish, OCR error)"
    )


class IsValidCitation(BaseModel):
    """
    A model to infer if a paper should be cited by a sentence.
    """

    is_valid: bool = Field(description="True if the paper should be cited by the input sentence, False otherwise")


class Findings(BaseModel):
    """
    A model to represent findings extracted from a scientific paper.
    """

    findings: list[str] = Field(description="List of original findings extracted from a scientific paper")


class LLMResponse(RootModel[str]):
    """
    The text requested by the user
    """
