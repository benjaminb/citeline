import os
import psycopg2
from psycopg2.extras import execute_values
import sys
import torch
from collections import namedtuple, defaultdict
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from semantic_text_splitter import TextSplitter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from time import time

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# fmt: off
from utils import load_dataset
from embedding_functions import get_embedding_fn 
# fmt: on

PGVECTOR_DISTANCE_METRICS = {
    'vector_l2_ops': '<->',
    'vector_ip_ops': '<#>',
    'vector_cosine_ops': '<=>',
}


class DatabaseProcessor:
    def __init__(self, db_params):
        self.db_params = db_params
        self.device = 'cuda' if torch.cuda.is_available(
        ) else 'mps' if torch.mps.is_available() else 'cpu'

    def __remove_nul_chars(self, s: str) -> str:
        """Remove NUL characters from a string, which PostgreSQL does not like"""
        return s.replace('\x00', '')

    def _insert_chunk(self, text: str, doi: str):
        text = self.__remove_nul_chars(text)
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chunks (text, doi) VALUES (%s, %s);", (text, doi))
        conn.commit()
        cursor.close()
        conn.close()

    def _chunk_record(self, record: dict, max_length: int, overlap: int):
        splitter = TextSplitter(capacity=max_length, overlap=overlap)
        full_text = record['title'] + '\n\nABSTRACT:\n' + \
            record['abstract'] + '\n\n' + record['body']
        chunks = splitter.chunks(full_text)
        doi = record['doi'][0]
        return [(chunk, doi) for chunk in chunks]

    def chunk_and_insert_records(self, path: str, max_length: int = 1500, overlap: int = 150):
        records = load_dataset(path)
        records = records[:50]  # For testing purposes
        all_chunks = []

        # Use ProcessPoolExecutor to parallelize chunking
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor:
            futures = [process_executor.submit(
                self._chunk_record, record, max_length, overlap) for record in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunking records"):
                all_chunks.extend(future.result())  # Collect all chunks

        # Use ThreadPoolExecutor to parallelize inserting chunks
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor:
            futures = [thread_executor.submit(
                self._insert_chunk, chunk, doi) for chunk, doi in all_chunks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Inserting chunks"):
                future.result()  # This will raise any exceptions caught during processing

    def create_vector_table(self, name, dim, embedder):
        """
        1. Creates a vector table 
        2. Creates indexes for all distance metrics
        3. Batch embeds all records in the `chunks` table using the given embedder

        available distance metrics:
        vector_l2_ops, vector_ip_ops, vector_cosine_ops, vector_l1_ops, bit_hamming_ops, bit_jaccard_ops
        """
        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()
        cursor.execute(
            f"""CREATE TABLE {name} (
                id SERIAL PRIMARY KEY, 
                embedding VECTOR({dim}), 
                chunk_id INTEGER REFERENCES chunks(id)
                );
            """)
        conn.commit()
        print(f"Created table {name}")

        # Create indexes
        for metric in PGVECTOR_DISTANCE_METRICS:
            cursor.execute(
                f"CREATE INDEX ON {name} USING hnsw (embedding {metric})")
        conn.commit()

        # Get all chunks for embedding
        ids_and_chunks = self._get_all_chunks(cursor)
        print(f"Embedding {len(ids_and_chunks)} chunks...")

        # Embed an insert in batches
        batch_size = 16
        num_batches = len(ids_and_chunks) // batch_size
        for i in tqdm(range(num_batches), desc="Inserting embeddings", leave=False):
            # Prepare batch
            batch = ids_and_chunks[i * batch_size:(i + 1) * batch_size]
            ids, texts = list(zip(*batch))
            embeddings = embedder(texts)
            data = [(embedding, id_num)
                    for embedding, id_num in zip(embeddings, ids)]

            # Insert
            execute_values(
                cursor, f"INSERT INTO {name} (embedding, chunk_id) VALUES %s;", data)
            conn.commit()
        cursor.close()

    def _get_all_chunks(self, cursor, columns: list[str] = ['id', 'text']) -> list[dict]:
        cursor.execute(f"SELECT id, text FROM chunks;")
        return cursor.fetchall()

    def query_vector_table(self, table_name, query_vector, metric, top_k=5):
        """
        table_name: name of the vector table
        query_vector: the vector to query
        metric: a key in PGVECTOR_DISTANCE_METRICS to resolve the distance operator
        top_k: number of results to return

        """

        # Resolve the distance operator
        assert metric in PGVECTOR_DISTANCE_METRICS, f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"
        operator = PGVECTOR_DISTANCE_METRICS[metric]

        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {table_name}.chunk_id, chunks.doi, chunks.text, {table_name}.embedding {operator} %s AS distance 
            FROM {table_name} 
            JOIN chunks ON {table_name}.chunk_id = chunks.id
            ORDER BY embedding {operator} %s DESC 
            LIMIT %s;
            """,
            (query_vector, query_vector, top_k)
        )
        results = cursor.fetchall()
        cursor.close()

        # Define the named tuple
        QueryResult = namedtuple(
            'QueryResult', ['chunk_id', 'doi', 'text', 'distance'])
        return [QueryResult(*result) for result in results]

    def test_connection(self):
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        # Execute a simple query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"Database version: {db_version}")

        cursor.close()
        conn.close()


def main():
    load_dotenv()
    db_params = {
        'dbname': 'test',
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }

    processor = DatabaseProcessor(db_params)
    processor.test_connection()
    print(processor.db_params)

    from embedding_functions import get_embedding_fn
    embedder = get_embedding_fn(
        'BAAI/bge-small-en', processor.device, normalize=False)

    start = time()

    # results = processor.query_vector_table('test_bge', vector)

    processor.create_vector_table(
        'test_bge',
        dim=384,
        embedder=embedder)
    end = time() - start


if __name__ == '__main__':
    main()
