
"""
An 'enrichment function' is a function
(example, parent record) -> text


then whatever the enrichment function is, in the class we have a method that takes the example, 
resolves the record, then calls the enrichment function with the example and the record
"""
import pandas as pd


class Enricher:
    def __init__(self, enrichment_function, data):
        self.enrichment_function = enrichment_function
        self.data = data

    def enrich(self, example):
        record = self.__get_record_by_doi(example.source_doi)
        if record is None:
            raise ValueError(
                f"Record with DOI {example.source_doi} not found to match example:\n{example}")
        print(f"Record datatype: {type(record)}")
        print(f"Record found: {record['title']}")
        return self.enrichment_function(example, record)

    def __get_record_by_doi(self, doi):
        matching_row = self.data[self.data['doi'].apply(lambda x: doi in x)]
        if not matching_row.empty:
            return matching_row.iloc[0]
        return None


def add_abstract(example, record):
    print(f"Keys on record: {record.keys()}")
    return record['abstract'] + '\n' + example.sent_no_cit


def add_title(example, record):
    return record['title'] + '\n' + example.sent_no_cit


def add_title_and_abstract(example, record):
    return record['title'] + '\n' + record['abstract'] + '\n' + example.sent_no_cit


def add_previous_3_sentences(example, record):
    sent_idx = example.sent_idx
    prev_3 = record['body_sentences'][sent_idx-3:sent_idx]
    return ' '.join(prev_3) + '\n' + example.sent_no_cit


def no_augmentation(example, record):
    return example.sent_no_cit


def main():

    from preprocessing import load_dataset

    data = pd.DataFrame(load_dataset(
        'data/sentence_segmented_and_merged/Astro_Reviews.json'))
    enricher = Enricher(add_previous_3_sentences, data)

    # Load nontrivial examples
    examples = pd.read_json(
        'data/dataset/no_reviews/nontrivial.jsonl', lines=True)

    for i, example in examples.iterrows():
        print(f"Example source doi: {example.source_doi}")
        enriched_text = enricher.enrich(example)
        print(f"Original sentence: {example.sent_no_cit}")
        print(f"Enriched sentence: {enriched_text}")
        if i > 5:
            break


if __name__ == "__main__":
    main()
