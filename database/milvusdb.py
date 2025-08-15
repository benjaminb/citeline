import argparse
from pymilvus import MilvusClient, Collection, FieldSchema, CollectionSchema, DataType, utility
import os
from dotenv import load_dotenv
from tqdm import tqdm
from embedders import get_embedder, Embedder
import torch
import pandas as pd

load_dotenv("../.env")


def argument_parser():
    parser = argparse.ArgumentParser(description="Milvus DB Management")

    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument("--drop-collection", type=str, help="Name of the collection to drop")
    operation_group.add_argument("--create-collection", action="store_true", help="Create a new collection")
    operation_group.add_argument("--list-collections", action="store_true", help="List all collections")
    operation_group.add_argument("--healthcheck", action="store_true", help="Check the health of the Milvus server")

    # Arguments required when creating a collection
    parser.add_argument("--name", type=str, help="Name to give the new collection")
    parser.add_argument("--data-source", type=str, help="Path to the JSONL data file")
    parser.add_argument("--embedder", type=str, help="Name of the embedder (e.g. 'BAAI/bge-en-large-v1.5')")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the embeddings")

    args = parser.parse_args()

    # Validate required arguments for create-collection
    if args.create_collection:
        if not all([args.name, args.data_source, args.embedder]):
            parser.error("--create-collection requires --name, --data-source, and --embedder arguments")

    return args


class MilvusDB:
    BASE_FIELDS = [
        {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
        {"field_name": "text", "datatype": DataType.VARCHAR, "max_length": 2048},
        {"field_name": "doi", "datatype": DataType.VARCHAR, "max_length": 64},
        {"field_name": "pubdate", "datatype": DataType.INT64},  # Milvus has no date type
    ]

    CLEAR_GPU_CACHE_FN = {"cuda": torch.cuda.empty_cache, "mps": torch.mps.empty_cache, "cpu": lambda: None}

    def __init__(self, alias: str = "default"):
        self.client = MilvusClient(alias=alias)

        # Set device and its related clear cache function
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.clear_gpu_cache = self.CLEAR_GPU_CACHE_FN[self.device]

    def healthcheck(self):
        status = self.client.get_server_version()
        print(f"Milvus server version: {status}")

    def create_vector_collection(self, name: str, data_source: str, embedder_name: str, normalize: bool):
        """
        Creates a collection using the base fields and a single vector field inferred by the embedder.

        Note that all collections will have nearly identical schema:
            vector: the vector representation of the document
            text: the text content of the document (original text, contribution, or other document expansion)
            doi: the DOI of the research paper from which the text originates
            pubdate: the publication date of the research paper (int YYYYMMDD format)
        Args:
            name: Name of the collection to create
            data_source: path to a jsonl file with entities containing keys 'text', 'doi', and 'pubdate' (int YYYYMMDD format)
            embedder_name: Name of the embedder to use for generating vector embeddings
            normalize: Whether to normalize the embeddings
        """
        if name in self.client.list_collections():
            print(f"Collection '{name}' already exists.")
            return

        # Make sure we have CPU count available
        assert "CPUS" in os.environ, "CPUS environment variable not set."
        try:
            num_cpus = int(os.getenv("CPUS"))
        except ValueError:
            raise ValueError(f"Invalid value for CPUS environment variable.")

        # Load in data and embedder to be used
        data = pd.read_json(data_source, lines=True)
        assert set(data.columns) == {
            "text",
            "doi",
            "pubdate",
        }, "DataFrame must contain 'text', 'doi', and 'pubdate' columns (and no others)."
        embedder = get_embedder(embedder_name, device=self.device, normalize=normalize)

        # Print a table of the collection to be created, num cpus, embedder name and its dimension
        print(f"| {'Collection Name':<20} | {'Num CPUs':<10} | {'Embedder Name':<25} | {'Embedder Dim':<12} |")
        print(f"| {name:<20} | {num_cpus:<10} | {embedder_name:<25} | {embedder.dim:<12} |")

        # Set up the schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        vector_field = {"field_name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": embedder.dim}
        for field in self.BASE_FIELDS + [vector_field]:
            schema.add_field(**field)

        self.client.create_collection(collection_name=name, schema=schema)
        print(f"Collection '{name}' created")
        collection = Collection(name)
        self.__embed_and_insert(collection=collection, embedder=embedder, data=data, num_cpus=num_cpus)

        # Create Milvus-required index on vector column
        collection.create_index(field_name="vector", index_params={"index_type": "FLAT", "metric_type": "IP"})

    def drop_collection(self, name: str):
        if self.client.has_collection(name):
            self.client.drop_collection(name)
            print(f"Collection '{name}' dropped.")
        else:
            print(f"Collection '{name}' does not exist.")

    def list_collections(self):
        collections = self.client.list_collections()
        print("Collections:")
        for col in collections:
            print(f" - {col}")

    def __embed_and_insert(
        self,
        collection: Collection,
        embedder: Embedder,
        data: pd.DataFrame,
        num_cpus: int,
        batch_size: int = 16,
    ):
        """

        Args:
            collection: The collection to add the vector field to.
            embedder: The embedder to use for generating vector embeddings.
            data: DataFrame assumed to have columns ["text", "doi", "pubdate" (int YYYYMMDD format)]
            num_cpus: The number of CPU cores to use for parallel processing.
            batch_size: The batch size to use for embedding and inserting data.
        """
        from concurrent.futures import ThreadPoolExecutor
        import queue
        import threading

        insert_queue = queue.Queue(maxsize=num_cpus * 2)
        insertion_lock = threading.Lock()

        num_entities = len(data)
        data["vector"] = None

        def insert_worker():
            while True:
                try:
                    # Get batch and check if the queue is empty
                    batch_records = insert_queue.get(timeout=30)
                    if batch_records is None:
                        break

                    # Insert batch, update progress bar
                    collection.insert(batch_records)
                    with insertion_lock:
                        insert_bar.update(len(batch_records))

                except queue.Empty:
                    print("Insertion worker timed out waiting for data.")
                    break
                except Exception as e:
                    print(f"Insertion worker encountered an error: {e}")
                finally:
                    insert_queue.task_done()

        num_workers = max(1, num_cpus - 1)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                thread = executor.submit(insert_worker)

            # Create progress bars
            with tqdm(total=num_entities, desc="Embedding", unit="docs", position=0) as embed_bar, tqdm(
                total=num_entities, desc="Inserting", unit="docs", position=1
            ) as insert_bar:

                # Main thread: producer (embed using gpu)
                clear_cache_interval = 50 * batch_size  # Will clear GPU cache every 50 batches
                for i in range(0, num_entities, batch_size):
                    if i % clear_cache_interval == 0:
                        self.clear_gpu_cache()

                    batch = data.iloc[slice(i, i + batch_size)]
                    embeddings = embedder(batch["text"])

                    batch_records = batch.to_dict(orient="records")
                    for record, vector in zip(batch_records, embeddings):
                        record["vector"] = vector
                    insert_queue.put(batch_records)

                    embed_bar.update(len(batch))

                for _ in range(num_workers):
                    insert_queue.put(None)

                insert_queue.join()

        print("All insertions complete.")
        print("Flushing to disk...")
        collection.flush()
        collection.create_index(field_name="vector", index_params={"index_type": "FLAT", "metric_type": "IP"})
        print(f"New collection {collection.name}: {collection.num_entities} entities")


def main():
    args = argument_parser()

    db = MilvusDB()

    if args.drop_collection:
        db.drop_collection(args.drop_collection)
    elif args.create_collection:
        db.create_vector_collection(
            name=args.name, data_source=args.data_source, embedder_name=args.embedder, normalize=args.normalize
        )
    elif args.list_collections:
        db.list_collections()
    elif args.healthcheck:
        db.healthcheck()


if __name__ == "__main__":
    main()
    # db = MilvusDB()
    # db.client.list_collections()
    # data = "../testdata.jsonl"
    # db.create_vector_collection(
    #     name="test_collection", data_source=data, embedder_name="BAAI/bge-small-en", normalize=False
    # )
    # db.client.list_collections()

    # import numpy as np

    # query = np.random.rand(384).astype(np.float32)
    # db.client.load_collection("test_collection")
    # res = db.client.search(
    #     collection_name="test_collection",
    #     anns_field="vector",
    #     data=[query],
    #     output_fields=["text", "doi", "pubdate"],
    #     limit=3,
    #     search_params={"metric_type": "IP"},
    # )
    # print(f"Search results:")
    # from pprint import pprint

    # pprint(res[0])

    # db.drop_collection("test_collection")
