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


# TODO: figure out how to handle these wrt reference chunks


def enricher_factory(keys_and_headers: list[tuple[str, str]], prev_n: int = 0):
    """
    Factory function to create an enrichment function based on the keys and number of previous sentences
    """

    # NOTE: changing this to take an example (pd.Series) rather than just the text
    def enrichment_function(example, record: pd.Series | dict) -> str:

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
            header_text = "\n\n".join(
                [f"{header}\n{record[key]}" for key, header in keys_and_headers]
            )
            text = f"IMPORTANT:\n{text}\n\n{header_text}" if header_text else text

        return text

    return enrichment_function


class TextEnricher:
    ENRICHMENT_FN = {
        "identity": enricher_factory([]),
        "add_abstract": enricher_factory([("abstract", "Abstract:")]),
        "add_title": enricher_factory([("title", "Title:")]),
        "add_title_and_abstract": enricher_factory(
            [("title", "Title:"), ("abstract", "Abstract:")]
        ),
        "add_prev_3": enricher_factory([], prev_n=3),
        "add_abstract_prev_3": enricher_factory([("abstract", "Abstract:")], prev_n=3),
        "add_title_prev_3": enricher_factory([("title", "Title:")], prev_n=3),
        "add_title_abstract_prev3": enricher_factory(
            [("title", "Title:"), ("abstract", "Abstract:")], prev_n=3
        ),
        "add_prev_7": enricher_factory([], prev_n=7),
    }

    def __init__(
        self,
        enrichment_function: str,
        reference_data: None = pd.DataFrame,
        # query_data: None = pd.DataFrame,
    ):
        """
        enrichment_function: function
            Takes an example and a record, and returns the enriched text.
        query_data: pd.DataFrame
            DataFrame containing the records to query against. These are the Reviews datasets
            and are used to enrich examples during evaluation or query time.
        reference_data: pd.DataFrame
            DataFrame containing the reference records. These are all the non-Reviews datasets
            and are used to enrich 'chunks' when creating enriched tables on the database
        """
        if not enrichment_function in self.ENRICHMENT_FN:
            raise KeyError(
                f"Enrichment function {enrichment_function} not supported. Available functions: {list(self.ENRICHMENT_FN.keys())}"
            )

        self.name = enrichment_function
        self.enricher = self.ENRICHMENT_FN[enrichment_function]
        self.doi_to_record = reference_data.set_index("doi").to_dict(orient="index")

    def __str__(self):
        return f"TextEnricher(name={self.name}, data_length={len(self.doi_to_record)})"

    def enrich_batch(self, texts_with_dois: list[tuple[str, str]]) -> list[str]:
        """
        Process multiple texts with their associated DOIs in one batch

        Args:
            texts_with_dois (list of tuples): Each tuple contains a text string and its corresponding DOI. (text, doi)
        """
        results = []

        for text, doi in texts_with_dois:
            record = self.doi_to_record.get(doi)
            if record is None:
                raise ValueError(
                    f"While enriching example with source doi '{doi}', full record not found"
                )
            results.append(self.enricher(text, record))

        return results

    def __call__(self, examples: pd.DataFrame) -> list[str]:
        results = []
        # for _, example in examples.iterrows():
        for example in examples.itertuples():
            # Get the full record for this example
            record = self.doi_to_record.get(example.source_doi)

            if record is None:
                raise ValueError(
                    f"While enriching example with source doi '{example.source_doi}', full record not found"
                )
            results.append(self.enricher(example, record))
        return results


def get_enricher(name: str, path_to_data: str) -> TextEnricher:
    try:
        data = pd.read_json(path_to_data, lines=True)
        return TextEnricher(enrichment_function=name, reference_data=data)
    except Exception as e:
        print(f"Error loading data source: {e}")


def main():

    enricher = get_enricher("identity", for_query=False)

    examples = pd.read_json("data/dataset/small/nontrivial.jsonl", lines=True)
    batch = [(example.sent_no_cit, example.source_doi) for _, example in examples.iterrows()]
    batch = batch[:2]

    enriched_batch = enricher.enrich_batch(batch)
    for sample in enriched_batch[:]:
        print(sample)
        print("===" * 20)


if __name__ == "__main__":
    main()
