import os
import psycopg2
from psycopg2.extras import execute_values
import sys
import torch
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

    def vectorize_chunk(self, text: str, embedder):
        return embedder([text])

    def create_vector_table(self, name, dim, embedder):
        # self.embedder = embedder
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

        # TODO: parameterize the distance function?
        '''
        available distance metrics:
        vector_l2_ops, vector_ip_ops, vector_cosine_ops, vector_l1_ops, bit_hamming_ops, bit_jaccard_ops
        '''
        # Create index
        cursor.execute(
            f"CREATE INDEX ON {name} USING hnsw (embedding vector_cosine_ops)")

        ids_and_chunks = self._get_all_chunks(cursor)
        print(f"Embedding {len(ids_and_chunks)} chunks...")

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
        # return [{'id': result[0], 'text': result[1]} for result in results]

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

    # conn = psycopg2.connect(**db_params)
    # cursor = conn.cursor()
    start = time()
    # res = processor._get_all_chunks(cursor)
    # print(f"Result: {len(res)}")
    # print(f"First result: {res[0]}")
    # processor.chunk_and_insert_records('../data/json/Astro_Research.json')
    chunks = processor.create_vector_table(
        'test_bge', 384, embedder=embedder)
    end = time() - start

    print(f"Time taken: {end} seconds")


if __name__ == '__main__':
    main()
