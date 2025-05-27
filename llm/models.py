from pydantic import BaseModel, Field


class SentenceValidation(BaseModel):
    reasoning: str = Field(
        description="Brief reasoning for why the sentence is valid or not. Consider if it's natural language, has clear meaning, and is not a figure caption, table content, reference list item, or gibberish."
    )
    is_valid: bool = Field(
        description="True if the sentence is a 'good' scientific sentence (natural language, clear meaning), False otherwise (e.g., caption, reference, gibberish, OCR error)."
    )


class Citation(BaseModel):
    author: str = Field(description="First author of the cited work")
    year: str = Field(
        description="Publication year of the cited work, possibly with letters (e.g., '2023a')"
    )


class CitationList(BaseModel):
    citations: list[Citation] = Field(
        description="List of inline citations extracted from the sentence"
    )

class SentenceNoCitation(BaseModel):
    citations: CitationList = Field(
        description="List of inline citations extracted from the sentence, if any"
    )
    sentence: str = Field(
        description="A sentence with any inline citations replaced by '[REF]' placeholders"
    )


class CitationExtraction(BaseModel):
    citation_list: CitationList = Field(
        description="List of inline citations extracted from the sentence, if any"
    )
    sentence: str = Field(
        description="A sentence with any inline citations replaced by '[REF]' placeholders"
    )
