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

    parser.add_argument("--drop-collection", type=str, help="Name of the collection to drop")
    parser.add_argument("--create-collection", type=str, help="JSON string of fields to add")
    parser.add_argument("--add-vector-field", type=str)

    # If adding a vector field, we need the collection name and the name of the embedder to use
    parser.add_argument("--field-name", type=str, help="Name of the vector field to add")
    parser.add_argument("--to-collection", type=str, help="Name of the collection to add the field to")
    parser.add_argument("--embedder", type=str, help="Name of the embedder (e.g. 'BAAI/bge-en-large-v1.5')")

    return parser


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

    def _get_field_schemas(self, collection: Collection) -> list[FieldSchema]:
        fields = []
        for field in collection.schema.fields:
            new_field = FieldSchema(
                name=field.name,
                dtype=field.dtype,
                is_primary=field.is_primary,
                auto_id=field.auto_id,
                max_length=field.max_length if field.dtype == DataType.VARCHAR else None,
                dim=(
                    field.dim
                    if field.dtype in [DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR]
                    else None
                ),
                description=field.description,
            )
            fields.append(new_field)

        return fields

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

    def insert(self, collection_name: str, data: list[dict]) -> int:
        return self.client.insert(collection_name=collection_name, data=data)

    def add_fields_to_collection(self, collection_name: str, fields: list[FieldSchema]):
        """
        Takes an existing collection, creates a copy with new field schemas added,
        and replaces the old collection with the new one.
        """
        if not collection_name in self.client.list_collections():
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        # Get the existing collection
        collection = Collection(collection_name)

        # Create a new schema with the added fields
        new_fields = self._get_field_schemas(collection) + fields
        new_schema = CollectionSchema(
            fields=new_fields,
            description=collection.schema.description,
        )

        # Copy data from the old collection to the new one
        new_collection_name = f"{collection_name}_new"
        self.client.create_collection(name=new_collection_name, schema=new_schema)

        # Replace the old collection with the new one
        self.client.drop_collection(collection_name)
        self.client.create_collection(name=collection_name, schema=new_schema)

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
        data['vector'] = None
        
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

        insert_threads = []
        num_workers = max(1, num_cpus - 1)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                thread = executor.submit(insert_worker)
                insert_threads.append(thread)

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

                    batch_records = batch.to_dict(orient='records')
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


if __name__ == "__main__":
    db = MilvusDB()
    db.client.list_collections()
    data = "../testdata.jsonl"
    db.create_vector_collection(
        name="test_collection", data_source=data, embedder_name="BAAI/bge-small-en", normalize=False
    )
    db.client.list_collections()

    import numpy as np

    query = np.random.rand(384).astype(np.float32)
    db.client.load_collection("test_collection")
    res = db.client.search(
        collection_name="test_collection",
        anns_field="vector",
        data=[query],
        output_fields=["text", "doi", "pubdate"],
        limit=3,
        search_params={"metric_type": "IP"},
    )
    print(f"Search results:")
    from pprint import pprint

    pprint(res[0])

    db.drop_collection("test_collection")
