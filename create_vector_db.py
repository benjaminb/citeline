import argparse
import chromadb
import uuid
from embedding_functions import get_embedding_fn
from utils import load_dataset


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the dataset')
    parser.add_argument('--collection', type=str,
                        help='Name of the collection')
    parser.add_argument('--model', type=str,
                        help='Name of the embedding model')
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_dataset(args.path)

    print(f"Retrieved {len(data)} records")
    print(f"Each record has keys: {data[0].keys()}")
    exit()
    embedding_fn = get_embedding_fn(args.model)

    # Create a collection on the db
    client = chromadb.Client()
    collection = client.create_collection(
        name=args.collection,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Iterate over the dataset
    for record in data:
        # Get a vector embedding
        # vector = embedding_fn(document['text'])
        # HERE need a function that takes the record and a chunking strategy
        # the return object is a list of (doc_id, text chunk, vector) tuples ready for db insertion
        # Insert the document into the collection

        result = collection.add(
            documents=documents[:3],
            metadatas=metadatas[:3],
            ids=ids[:3]
        )


if __name__ == "__main__":
    main()
