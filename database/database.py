from collections import namedtuple
import argparse
import os
import psycopg2
from psycopg2.extras import execute_values
import sys
import torch
from collections import namedtuple
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from semantic_text_splitter import TextSplitter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# fmt: off
# NOTE: moved into chunk_and_insert since it's not used elsewhere
# from preprocessing import load_dataset

# fmt: on

MAX_CPU_CORES = 12
PGVECTOR_DISTANCE_METRICS = {
    'vector_l2_ops': '<->',
    'vector_ip_ops': '<#>',
    'vector_cosine_ops': '<=>',
}


def argument_parser():
    parser = argparse.ArgumentParser(
        description='Database operations'
    )
    parser.add_argument(
        '--create-vector-table', '-v',
        type=str,
        help='Create a new vector table with the specified name'
    )
    parser.add_argument(
        '--embedder', '-e',
        type=str,
        default='BAAI/bge-small-en',
        help='Name of the embedding model to use (default: BAAI/bge-small-en)'
    )
    parser.add_argument(
        '--normalize', '-n',
        default=False,
        action='store_true',
        help='Normalize embeddings before inserting into the database'
    )
    parser.add_argument(
        '--add-chunks', '-a',
        type=str,
        help='Add chunks to the database from raw json files'
    )
    return parser.parse_args()


"""
Helper functions for vector table insertion. These must be outside the class definition to be pickled,
which is necessary for multiprocessing.
"""


def insert_batch(data_name_and_processor):
    data, name, processor = data_name_and_processor
    conn = psycopg2.connect(**processor.db_params)
    cursor = conn.cursor()
    execute_values(
        cursor,
        f"INSERT INTO {name} (embedding, chunk_id) VALUES %s;",
        data
    )
    conn.commit()
    cursor.close()
    conn.close()


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
        from preprocessing import load_dataset
        records = load_dataset(path)
        all_chunks = []

        # Use ProcessPoolExecutor to parallelize chunking
        cores = min(os.cpu_count(), MAX_CPU_CORES)
        with ProcessPoolExecutor(max_workers=cores) as process_executor:
            futures = [process_executor.submit(
                self._chunk_record, record, max_length, overlap) for record in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunking records"):
                all_chunks.extend(future.result())  # Collect all chunks

        # Use ThreadPoolExecutor to parallelize inserting chunks
        chunk_count = 0
        with ThreadPoolExecutor(max_workers=cores) as thread_executor:
            with tqdm(total=len(all_chunks), desc="Inserting chunks") as pbar:
                futures = [thread_executor.submit(
                    self._insert_chunk, chunk, doi) for chunk, doi in all_chunks]
                for future in as_completed(futures):
                    future.result()  # This will raise any exceptions caught during processing
                    chunk_count += 1
                    pbar.update(1)
                    if chunk_count % 100 == 0:
                        print(
                            f"\rInserted {chunk_count}/{len(all_chunks)} chunks", end="")

    def create_vector_table_mp(self, name, dim, embedder):
        """
        1. Creates a vector table 
        2. Creates indexes for all distance metrics
        3. Batch embeds all records in the `chunks` table using the given embedder
        """
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
            print(f"Creating index on {metric}...", end="")
            cursor.execute(
                f"CREATE INDEX ON {name} USING hnsw (embedding {metric})")
            print("done.")
        conn.commit()

        # Get all chunks for embedding
        ids_and_chunks = self._get_all_chunks(cursor)
        print(f"Embedding {len(ids_and_chunks)} chunks...")

        # Process in larger batches for GPU efficiency
        batch_size = 512
        num_batches = len(ids_and_chunks) // batch_size + 1

        for i in tqdm(range(num_batches), desc="Processing batches"):
            # Prepare batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(ids_and_chunks))
            batch = ids_and_chunks[start_idx:end_idx]

            if not batch:
                continue

            ids, texts = list(zip(*batch))
            # Embeddings are automatically moved to CPU
            embeddings = embedder(texts)

            # Insert using ThreadPoolExecutor for parallel DB operations
            data = [(emb, id_) for emb, id_ in zip(embeddings, ids)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Split data for parallel insertion
                chunk_size = len(data) // 4
                data_chunks = [data[j:j + chunk_size]
                               for j in range(0, len(data), chunk_size)]

                futures = [
                    executor.submit(insert_batch, (chunk, name, self))
                    for chunk in data_chunks if chunk
                ]
                list(tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Inserting batch",
                    leave=False
                ))
        cursor.close()
        conn.close()

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
        batch_size = 64
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

    def batch_query_vector_table(self, table_name, query_vectors, metric, top_k=5):
        """
        TODO: this isn't working, figure out why
        table_name: name of the vector table
        query_vectors: list of vectors to query
        metric: a key in PGVECTOR_DISTANCE_METRICS to resolve the distance operator
        top_k: number of results to return for each query vector

        Returns a list of lists, where each inner list contains the results for one query vector
        """

        # Resolve the distance operator
        assert metric in PGVECTOR_DISTANCE_METRICS, f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"
        operator = PGVECTOR_DISTANCE_METRICS[metric]

        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()

        # Prepare the query
        query = f"""
        WITH query_vectors AS (
            SELECT * FROM unnest(%s::vector[]) WITH ORDINALITY AS t(query_vector, query_index)
        )
        SELECT 
            qv.query_index,
            {table_name}.chunk_id, 
            chunks.doi, 
            chunks.text, 
            {table_name}.embedding {operator} qv.query_vector AS distance
        FROM query_vectors qv
        CROSS JOIN LATERAL (
            SELECT {table_name}.chunk_id, chunks.doi, chunks.text, {table_name}.embedding
            FROM {table_name}
            JOIN chunks ON {table_name}.chunk_id = chunks.id
            ORDER BY {table_name}.embedding {operator} qv.query_vector
            LIMIT %s
        ) AS results
        ORDER BY qv.query_index, distance DESC
        """

        # Execute the query
        cursor.execute(query, (query_vectors, top_k))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        # Define the named tuple
        VectorQueryResult = namedtuple(
            'VectorQueryResult', ['chunk_id', 'doi', 'text', 'similarity'])

        # Group results by query_index
        grouped_results = {}
        for result in results:
            query_index = result[0]
            if query_index not in grouped_results:
                grouped_results[query_index] = []
            grouped_results[query_index].append(VectorQueryResult(*result[1:]))

        # Return results as a list of lists, maintaining the order of input query vectors
        return [grouped_results.get(i+1, []) for i in range(len(query_vectors))]

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
        VectorQueryResult = namedtuple(
            'VectorQueryResult', ['chunk_id', 'doi', 'text', 'similarity'])
        return [VectorQueryResult(*result) for result in results]

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
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }

    db = DatabaseProcessor(db_params)
    db.test_connection()
    print(db.db_params)

    args = argument_parser()

    if args.add_chunks:
        db.chunk_and_insert_records(args.add_chunks)
        return

    if args.create_vector_table:

        # Extract parameters
        table_name, embedder, normalize = args.create_vector_table, args.embedder, args.normalize

        # Create embedding function and get its dimension
        from Embedders import get_embedder
        embedder = get_embedder(embedder, db.device, normalize)
        dim = embedder(['test']).shape[1]

        db.create_vector_table(
            name=table_name, dim=dim, embedder=embedder)
        return


if __name__ == '__main__':
    main()
