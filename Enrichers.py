
"""
An 'enrichment function' is a function
(example, parent record) -> text


then whatever the enrichment function is, in the class we have a method that takes the example, 
resolves the record, then calls the enrichment function with the example and the record
"""
import pandas as pd


class Enricher:
    def __init__(self, enrichment_function, query_data: None=pd.DataFrame, reference_data: None=pd.DataFrame):
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
        self.enrichment_function = enrichment_function
        self.data = query_data
        self.reference_data = reference_data

    def enrich_chunk(self, chunk):
        # get the chunk doi
        doi = chunk.doi

        # find the related row with that doi in reference_data
        index = self.reference_data[self.reference_data['doi'][0] == doi].iloc[0]
        # If no index found, raise an error
        if index.empty:
            raise ValueError(f"No matching record found for DOI: {doi}")        
        reference_record = self.reference_data.iloc[index]


        record = self.reference_data.iloc[index[0]]
        # then call the enrichment function with the chunk and the record

    def enrich(self, example):
        record = self.__get_record_by_doi(example.source_doi)
        if record is None:
            raise ValueError(
                f"Record with DOI {example.source_doi} not found to match example:\n{example}")
        return self.enrichment_function(example, record)
    
    def enrich_batch(self, examples):
        return [self.enrich(example) for _, example in examples.iterrows()]

    def __get_record_by_doi(self, doi):
        matching_row = self.data[self.data['doi'].apply(lambda x: doi in x)]
        if not matching_row.empty:
            return matching_row.iloc[0]
        return None


def add_abstract(example, record):
    return record['abstract'] + '\n' + example.sent_no_cit


def add_title(example, record):
    return record['title'] + '\n' + example.sent_no_cit


def add_title_and_abstract(example, record):
    return record['title'] + '\n' + record['abstract'] + '\n' + example.sent_no_cit


def add_previous_3_sentences(example, record):
    sent_idx = example.sent_idx
    start_idx = max(sent_idx-3, 0)
    prev_3 = record['body_sentences'][start_idx:sent_idx]
    return ' '.join(prev_3) + '\n' + example.sent_no_cit

def add_previous_7_sentences(example, record):
    sent_idx = example.sent_idx
    start_idx = max(sent_idx-7, 0)
    prev_7 = record['body_sentences'][start_idx:sent_idx]
    return ' '.join(prev_7) + '\n' + example.sent_no_cit

def add_headers_and_previous_3_sentences(example, record):
    sent_idx = example.sent_idx
    start_idx = max(sent_idx-3, 0)
    prev_3 = record['body_sentences'][start_idx:sent_idx]
    title = record['title']
    abstract = record['abstract']
    return title + '\n' + abstract + '\n' + ' '.join(prev_3) + '\n' + example.sent_no_cit

def add_headers_previous_7_sentences(example, record):
    sent_idx = example.sent_idx
    start_idx = max(sent_idx-7, 0)
    prev_7 = record['body_sentences'][start_idx:sent_idx]
    title = record['title']
    abstract = record['abstract']
    return title + '\n' + abstract + '\n' + ' '.join(prev_7) + '\n' + example.sent_no_cit

def no_augmentation(example, record):
    return example.sent_no_cit


ENRICHMENT_FN = {
    'identity': no_augmentation,
    'add_abstract': add_abstract,
    'add_title': add_title,
    'add_title_and_abstract': add_title_and_abstract,
    'add_previous_3_sentences': add_previous_3_sentences,
    'add_previous_7_sentences': add_previous_7_sentences,
    'add_headers_and_previous_3_sentences': add_headers_and_previous_3_sentences,
    'add_headers_and_previous_7_sentences': add_headers_previous_7_sentences
}


# def get_enricher(name: str, data):
def get_enricher(name: str, for_query: bool=True) -> Enricher:
    try:
        if for_query:
            json_files = [
                'data/preprocessed/Astro_Reviews.json',
                'data/preprocessed/Earth_Science_Reviews.json',
                'data/preprocessed/Planetary_Reviews.json'
            ]
            dfs = [pd.read_json(file) for file in json_files]
            data = pd.concat(dfs, ignore_index=True)
            return Enricher(ENRICHMENT_FN[name], query_data=data)
        else:
            data = pd.read_json('data/preprocessed/research.jsonl', lines=True)
            return Enricher(ENRICHMENT_FN[name], reference_data=data)
    except KeyError:
        raise ValueError(
            f"Enrichment function {name} not supported. Available functions: {list(ENRICHMENT_FN.keys())}")


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
