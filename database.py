import argparse
import chromadb
import pysbd
import torch
import uuid
from tqdm import tqdm
from embedding_functions import get_embedding_fn
from utils import load_dataset
from sentence_validators import length_at_least_40

SEG = pysbd.Segmenter(language="en", clean=False)
METRICS = ['cosine', 'l2', 'ip']
DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.mps.is_available() else 'cpu'


def parse_args():
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Manage collections in the database")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Create the parser for the "delete-collection" command
    delete_parser = subparsers.add_parser(
        "delete-collection", help="Delete a collection")
    delete_parser.add_argument(
        "--name", type=str, required=True, help="Name of the collection to delete")

    # Create the parser for the "create-collection" command
    create_parser = subparsers.add_parser(
        "create-collection", help="Create a collection")
    create_parser.add_argument(
        "--model", type=str, required=True, help="Name of the embedding model")
    create_parser.add_argument("--metric", type=str, required=True,
                               choices=["l2", "cosine", "ip"], help="'l2', 'cosine', or 'ip'")
    create_parser.add_argument(
        "--augment-fn", type=str, required=True, help="Augmentation function")

    insert_parser = subparsers.add_parser(
        "insert-records", help="Insert records into a collection")
    insert_parser.add_argument(
        "--into", type=str, required=True, help="Name of the collection")
    insert_parser.add_argument(
        "--path", type=str, required=True, help="Path to the dataset")

    list_parser = subparsers.add_parser(
        "list-collections", help="List all collections")

    return parser.parse_args()


def augmented_sentence_embeddings(record, embedding_fn, augmenting_fn, is_valid_sentence):
    """
    Takes a record's body, segments it into sentences, augments each sentence 
    for embedding (such as with title or abstract), and returns a list of tuples 
    of the form (sentence, embedding).

    Args:
        record (dict): The record containing the body text to be segmented.
        embedding_fn (callable): The function to generate embeddings for the text.
        augmenting_fn (callable): The function to augment the sentences for embedding.
        is_valid_sentence (callable): A function to validate if a sentence should be included.

    Returns:
        tuple: A tuple containing:
            - list of str: The list of valid sentences.
            - list of np.ndarray: The list of corresponding embeddings.
    """
    sentences = [s for s in SEG.segment(
        record['body']) if is_valid_sentence(s)]

    # Batch the sentences into groups of 16
    batch_size = 16
    texts, vectors = [], []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i+batch_size]
        batch_texts = augmenting_fn(record, batch_sentences)
        batch_vectors = embedding_fn(batch_texts)
        texts.extend(batch_texts)
        vectors.extend(batch_vectors)
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


def delete_collection(client, collection_name):
    try:
        client.delete_collection(collection_name)
        print(f"Deleted collection {collection_name}")
    except Exception as e:
        print(f"Failed to delete collection {collection_name}: {e}")


def insert_records(collection, records, embedding_fn, augmenting_fn):
    count = collection.count()

    # Insert records
    for record in tqdm(records):
        sentences, vectors = augmented_sentence_embeddings(
            record, embedding_fn, augmenting_fn, length_at_least_40)
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


def query_database(client: chromadb.Client, collection_name: str, queries: list[str], n_results: int, include: list[str]):
    """
    Queries the specified collection in the database and returns the results.

    Args:
        client (chromadb.Client): The ChromaDB client instance.
        collection_name (str): The name of the collection to query.
        query_texts (list[str]): A list of query texts to search for in the collection.
        n_results (int): The number of results to return for each query text.
        include (list[str]): A list of strings specifying which fields to include in the results.
            Possible values are "embeddings", "documents", and "metadatas".

    Returns:
        dict: A dictionary containing the query results. The structure of the dictionary
            depends on the fields specified in the `include` parameter.
    """
    collection = client.get_collection(collection_name)
    return collection.query(
        query_texts=queries,
        n_results=n_results,
        include=["embeddings", "documents", "metadatas"])


AUGMENTATION_FUNCTIONS = [no_augmentation, add_title,
                          add_abstract, add_title_and_abstract]
AUG_FN_DICT = {fn.__name__: fn for fn in AUGMENTATION_FUNCTIONS}
CHROMA_MODEL_NAME_TO_HF = {
    'bge-small-en': 'BAAI/bge-small-en',
    'bert-base-uncased': 'bert-base-uncased',
    'NV-Embed-v2': 'nvidia/NV-Embed-v2'
}


def main():
    args = parse_args()
    client = chromadb.PersistentClient(path='./vector_stores/foo/')

    if args.command == "delete-collection":
        delete_collection(client, args.name)
        return

    if args.command == "list-collections":
        collections = client.list_collections()
        if not collections:
            print("No collections found.")
        for collection in collections:
            print(f"{collection.name:<40} ({collection.count()} records)")
        return

    if args.command == "create-collection":

        embedding_fn = get_embedding_fn(
            model_name=args.model,
            device=DEVICE,
            normalize=False
        )
        metric = args.metric
        aug_fn = AUG_FN_DICT[args.augment_fn]

        # Create a collection on the db
        collection = create_collection(client, embedding_fn, metric, aug_fn)
        return

    if args.command == "insert-records":
        # Resolve collection, model, metric, and augmenting function references
        collection = client.get_collection(args.into)
        model_name, metric, augmenting_fn_name = get_expected_parameters_from_collection_name(
            collection.name)

        if model_name == 'NV-Embed-v2' and DEVICE == 'mps':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(
                f"Switching compute device to {device}, mps incompatible with NV-Embed-v2")
        else:
            device = DEVICE

        embedding_fn = get_embedding_fn(
            model_name=CHROMA_MODEL_NAME_TO_HF[model_name],
            device=device,
            normalize=False
        )
        aug_fn = AUG_FN_DICT[augmenting_fn_name]

        data = load_dataset(args.path)
        insert_records(collection, data[:2], embedding_fn, aug_fn)
        return


if __name__ == "__main__":
    main()
