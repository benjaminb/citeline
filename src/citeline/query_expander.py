"""
An 'enrichment function' is a function
(text, context) -> text
Where 'text' is a string you typically want to embed (either from reference data or a query) and 'context'
is additional data relating to the text. In this project, 'context' is the record (structured data of a
research paper) from which the text is derived.




then whatever the enrichment function is, in the class we have a method that takes the example,
resolves the record, then calls the enrichment function with the example and the record
"""

import pandas as pd


BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def query_expander_factory(keys_and_headers: list[tuple[str, str]], prev_n: int = 0, prefix: str = ""):
    """
    Factory function to create an expansion function based on the keys and number of previous sentences
    """

    # NOTE: changing this to take an example (pd.Series) rather than just the text
    def query_expansion_function(example: pd.Series, record: pd.Series | dict) -> str:
        """
        Args:
            example: pd.Series
                The example Series from data, containing fields like 'sent_no_cit', 'sent_idx', etc.
            record: pd.Series or dict
                The full record from which the example is derived, containing additional fields like 'title', 'abstract', etc.
        """

        text = example.sent_no_cit
        if prev_n > 0:
            # Find index of the text in the body_sentences
            start_index = max(0, example.sent_idx - prev_n)
            end_index = example.sent_idx

            # Append the sentence with no citation to the previous n sentences
            prev_n_sentences = record["body_sentences"][start_index:end_index]
            text = " ".join(prev_n_sentences) + " " + text

        # Add the header and corresponding value
        if keys_and_headers:
            header_text = "\n\n".join([f"{header}\n{record[key]}" for key, header in keys_and_headers])
            text = f"IMPORTANT:\n{text}\n\n{header_text}" if header_text else text

        return prefix + text

    return query_expansion_function


class QueryExpander:
    EXPANSION_FN = {
        "identity": query_expander_factory([]),
        "add_abstract": query_expander_factory([("abstract", "Abstract:")]),
        "add_title": query_expander_factory([("title", "Title:")]),
        "add_title_and_abstract": query_expander_factory([("title", "Title:"), ("abstract", "Abstract:")]),
        "add_prev_1": query_expander_factory([], prev_n=1),
        "add_prev_2": query_expander_factory([], prev_n=2),
        "add_prev_3": query_expander_factory([], prev_n=3),
        "add_prev_5": query_expander_factory([], prev_n=5),
        "add_abstract_prev_3": query_expander_factory([("abstract", "Abstract:")], prev_n=3),
        "add_title_prev_3": query_expander_factory([("title", "Title:")], prev_n=3),
        "add_title_abstract_prev3": query_expander_factory([("title", "Title:"), ("abstract", "Abstract:")], prev_n=3),
        "add_prev_7": query_expander_factory([], prev_n=7),
    }

    def __init__(
        self,
        expansion_function_name: str,
        reference_data: None = pd.DataFrame,
    ):
        """
        expansion_function: function
            Takes an example and a record, and returns the expanded text.
        query_data: pd.DataFrame
            DataFrame containing the records to query against. These are the Reviews datasets
            and are used to enrich examples during evaluation or query time.
        reference_data: pd.DataFrame
            DataFrame containing the reference records. These are all the non-Reviews datasets
            and are used to enrich 'chunks' when creating enriched tables on the database
        """
        if not expansion_function_name in self.EXPANSION_FN:
            raise KeyError(
                f"Expansion function {expansion_function_name} not supported. Available functions: {list(self.EXPANSION_FN.keys())}"
            )

        self.name = expansion_function_name
        self.enricher = self.EXPANSION_FN[expansion_function_name]
        self.doi_to_record = reference_data.set_index("doi").to_dict(orient="index")

    def __str__(self):
        return f"QueryExpander(name={self.name}, data_length={len(self.doi_to_record)})"

    def expand_batch(self, texts_with_dois: list[tuple[str, str]]) -> list[str]:
        """
        Process multiple texts with their associated DOIs in one batch

        Args:
            texts_with_dois (list of tuples): Each tuple contains a text string and its corresponding DOI. (text, doi)
        """
        results = []

        for text, doi in texts_with_dois:
            record = self.doi_to_record.get(doi)
            if record is None:
                raise ValueError(f"While enriching example with source doi '{doi}', full record not found")
            results.append(self.enricher(text, record))

        return results

    def __call__(self, examples: pd.DataFrame) -> list[str]:
        results = []
        for example in examples.itertuples():
            # Get the full record for this example
            record = self.doi_to_record.get(example.source_doi)

            if record is None:
                raise ValueError(
                    f"While enriching example with source doi '{example.source_doi}', full record not found"
                )
            results.append(self.enricher(example, record))
        return results


def get_expander(name: str, path_to_data: str) -> QueryExpander:
    """
    path_to_data: str
        Path to the JSONL file containing the reference data (the Reviews dataset).
    """
    try:
        data = pd.read_json(path_to_data, lines=True)
        return QueryExpander(expansion_function_name=name, reference_data=data)
    except Exception as e:
        print(f"Error loading data source: {e}")


def main():

    expander = get_expander("identity", path_to_data="data/preprocessed/reviews.jsonl")
    print(f"Expander created: {expander}")

    for key, value in expander.doi_to_record.items():
        print(f"DOI: {key}, value: {type(value)}")
        print(f"Keys in the value dict: {list(value.keys())}")
        print("===" * 20)
        break


if __name__ == "__main__":
    main()
