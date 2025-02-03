import argparse
import chromadb
import itertools
import pysbd
import torch
import uuid
from tqdm import tqdm
from embedding_functions import get_embedding_fn
from utils import load_dataset

SEG = pysbd.Segmenter(language="en", clean=False)
METRICS = ['cosine', 'l2', 'ip']
DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.mps.is_available() else 'cpu'


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the dataset')
    parser.add_argument('--model', type=str,
                        help='Name of the embedding model')
    return parser.parse_args()


def augmented_sentence_embeddings(record, embedding_fn, augmenting_fn):
    """
    Takes a record's body, segments it into sentences, augments each sentence 
    for embedding (such as with title or abstract), and returns a list of tuples 
    of the form (sentence, embedding).

    NOTE: Sentences less than 40 chars are filtered, as these are likely only
    legends, captions, etc. and not full sentences.
    """
    sentences = [s for s in SEG.segment(record['body']) if len(s) > 40]
    texts = augmenting_fn(record, sentences)
    vectors = embedding_fn(texts)
    return sentences, vectors


def add_abstract(record, sentences):
    abstract = record['abstract']
    return [abstract + '\n' + sentence for sentence in sentences]


def add_title(record, sentences):
    title = record['title']
    return [title + '\n' + sentence for sentence in sentences]


def add_title_and_abstract(record, sentences):
    title = record['title']
    abstract = record['abstract']
    return [title + '\n' + abstract + '\n' + sentence for sentence in sentences]


def no_augmentation(record, sentences):
    return sentences


def create_collection(client, embedding_fn, metric, augmenting_fn):
    # Set up collection name
    model_name = embedding_fn.model_name
    aug_fn = augmenting_fn.__name__
    collection_name = f"test-{model_name}__{metric}__{aug_fn}"

    # Create collection
    print(f"Creating collection: {collection_name}...", end="")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": metric})
    print("created.")
    return collection


def get_expected_parameters_from_collection_name(collection_name):
    """
    Parses the expected model name, metric, and augmenting function from the collection name.
    Allows us to assert that the correct functions are being used when inserting into db
    """
    parts = collection_name.split("__")
    # TODO: fix this for the real runs when 'test-' won't prepend each collection name
    model_name = parts[0][5:]
    metric = parts[1]
    augmenting_fn_name = parts[2]
    return model_name, metric, augmenting_fn_name


def insert_records(collection, records, embedding_fn, augmenting_fn):
    # Ensure the passed in embedding function and augmenting function match the collection's expected functions
    model_name, metric, augmenting_fn_name = get_expected_parameters_from_collection_name(
        collection.name)
    assert model_name == embedding_fn.model_name, f"Expected embedding model '{model_name}' for collection {collection.name} but got '{embedding_fn.model_name}'"
    assert augmenting_fn_name == augmenting_fn.__name__, f"Expected augmentation function '{augmenting_fn_name}' for {collection.name} but got '{augmenting_fn.__name__}'"

    count = collection.count()

    # Insert records
    for record in tqdm(records):
        sentences, vectors = augmented_sentence_embeddings(
            record, embedding_fn, augmenting_fn)
        ids = [str(uuid.uuid4()) for _ in sentences]
        doi = record['doi'][0]
        collection.add(
            documents=sentences,
            metadatas=[{'doi': doi}] * len(sentences),
            embeddings=vectors,
            ids=ids
        )

    print(
        f"Added {collection.count() - count} records to collection {collection.name}")


def main():
    args = parse_args()
    data = load_dataset(args.path)
    client = chromadb.PersistentClient(path='./vector_stores/foo/')
    embedding_fn = get_embedding_fn(
        model_name=args.model,
        device=DEVICE,
        normalize=False
    )
    augmenting_functions = [no_augmentation, add_title,
                            add_abstract, add_title_and_abstract]

    # Create a collection on the db
    client = chromadb.Client()
    collections = [create_collection(client, embedding_fn, metric, aug_fn)
                   for metric, aug_fn in itertools.product(METRICS, augmenting_functions)]

    # for aug_fn in augmenting_functions:
    #     sents_and_vecs = [augmented_sentence_embeddings(record, embedding_fn, aug_fn)
    #                       for record in data[:2]]
    #     for metric in METRICS:
    #         collection = create_collection(
    #             client, embedding_fn, metric, aug_fn)
    #         insert_records(collection, data[:2], embedding_fn, aug_fn)
    # Iterate over the dataset
    for collection in collections:
        aug_fn_name = collection.name.split("__")[-1]
        print(
            f"Working on collection {collection.name}...")
        # Get the index of the function in augementing_functions whose name matches aug_fn_name
        function_names = [fn.__name__ for fn in augmenting_functions]
        aug_fn = augmenting_functions[function_names.index(aug_fn_name)]
        print(f"Resolved augmenting function: {aug_fn_name} and ref {aug_fn}")

        insert_records(collection, data[:2], embedding_fn, aug_fn)


if __name__ == "__main__":
    main()
