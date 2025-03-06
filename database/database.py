import argparse
import gc
import json
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

from time import time

import cProfile
import pstats

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
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for embedding and insertion operations (default 32)'
    )
    parser.add_argument(
        '--create-index', '-i',
        action='store_true',
        help='Create an index on the specified table'
    )
    parser.add_argument(
        '--table', '-T',
        type=str,
        help='Name of the table to create an index on'
    )
    parser.add_argument(
        '--metric', '-m',
        type=str,
        help='Distance metric to use'
    )
    parser.add_argument(
        '--index-type', '-I',
        type=str,
        default='ivfflat',
        help='Type of index to create (default: ivfflat) or hnsw'
    )
    parser.add_argument(
        '--add-chunks', '-a',
        type=str,
        help='Add chunks to the database from raw json files'
    )
    parser.add_argument(
        '--test-connection', '-t',
        default=False,
        action='store_true',
        help='Test the database connection'
    )
    parser.add_argument(
        '--profile', '-p',
        default=False,
        action='store_true',
        help='Profile the create_vector_table function'
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

    def __clear_gpu_memory(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()

    def __remove_nul_chars(self, s: str) -> str:
        """Remove NUL characters from a string, which PostgreSQL does not like"""
        return s.replace('\x00', '')

    def _chunk_record(self, record: dict, max_length: int, overlap: int):
        splitter = TextSplitter(capacity=max_length, overlap=overlap)
        full_text = record['title'] + '\n\nABSTRACT:\n' + \
            record['abstract'] + '\n\n' + record['body']
        chunks = splitter.chunks(full_text)
        doi = record['doi'][0]
        return [(chunk, doi) for chunk in chunks]

    def _insert_chunk(data, table_name, db_params):
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        execute_values(
            cursor,
            f"INSERT INTO {table_name} (embedding, chunk_id) VALUES %s;",
            data
        )
        conn.commit()
        cursor.close()
        conn.close()

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

        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        try:
            batch_size = 1000  # Adjust this based on your needs
            for i in tqdm(range(0, len(all_chunks), batch_size), desc="Inserting chunks"):
                batch = all_chunks[i:i + batch_size]
                data = [(self.__remove_nul_chars(chunk), doi)
                        for chunk, doi in batch]
                execute_values(
                    cursor,
                    "INSERT INTO chunks (text, doi) VALUES %s;",
                    data
                )
                conn.commit()
        finally:
            cursor.close()
            conn.close()

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

    def create_vector_table(self, name, dim, embedder, work_mem='1024MB', maintenance_work_mem='2048MB', batch_size=32):
        """
        1. Creates a vector table
        2. Creates indexes for all distance metrics
        3. Batch embeds all records in the `chunks` table using the given embedder

        available distance metrics:
        vector_l2_ops, vector_ip_ops, vector_cosine_ops, vector_l1_ops, bit_hamming_ops, bit_jaccard_ops
        """

        # Create connection and allocate session memory
        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()
        cursor.execute(f"SET maintenance_work_mem='{maintenance_work_mem}';")
        cursor.execute(f"SET work_mem='{work_mem}';")

        # Create table
        cursor.execute(
            f"""CREATE TABLE {name} (
                id SERIAL PRIMARY KEY,
                embedding VECTOR({dim}),
                chunk_id INTEGER REFERENCES chunks(id)
                );
            """)
        conn.commit()
        print(f"Created table {name}")

        # Get all chunks for embedding
        ids_and_chunks = self._get_all_chunks(cursor)
        print(f"Embedding {len(ids_and_chunks)} chunks...")

        # Embed an insert in batches
        num_batches = 1 + len(ids_and_chunks) // batch_size
        for i in tqdm(range(num_batches), desc="Inserting embeddings", leave=False):
            # Clear GPU memory every 50 batches
            if i % 50 == 0:
                self.__clear_gpu_memory()

            # Prepare batch
            batch = ids_and_chunks[i * batch_size:(i + 1) * batch_size]
            ids, texts = list(zip(*batch))

            # Embed texts
            start_embeddings = time()
            embeddings = embedder(texts)
            print(f"Embedding time: {time() - start_embeddings}")

            data = [(embedding, id_num)
                    for embedding, id_num in zip(embeddings, ids)]

            # Insert
            start_insert = time()
            execute_values(
                cursor, f"INSERT INTO {name} (embedding, chunk_id) VALUES %s;", data)
            print(f"Insert time: {time() - start_insert}")
            conn.commit()

        cursor.close()

    def create_index(self,
                     table_name: str,
                     index_type: str,  # 'ivfflat' or 'hnsw'
                     metric: str,
                     num_lists: int = 1580,  # sqrt(num chunks, which is ~2.5M)
                     maintenance_work_mem='5GB',
                     max_parallel_maintenance_workers=4,
                     max_parallel_workers=4):
        assert index_type in [
            'ivfflat', 'hnsw'], f"Invalid index type: {index_type}. Must be 'ivfflat' or 'hnsw'"
        assert metric in PGVECTOR_DISTANCE_METRICS, f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        cursor.execute(f"SET maintenance_work_mem='{maintenance_work_mem}';", )
        cursor.execute(
            f"SET max_parallel_maintenance_workers={max_parallel_maintenance_workers};")
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")

        cursor.execute(
            f"CREATE INDEX ON {table_name} USING {index_type}(embedding {metric}) WITH (lists = {num_lists});")
        conn.commit()
        cursor.close()
        conn.close()

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

    def query_vector_table(self,
                           table_name,
                           query_vector,
                           metric,
                           top_k=5,
                           probes=40,
                           work_mem='2GB',
                           max_parallel_workers=8,
                           max_parallel_workers_per_gather=8):
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

        # Set the session resources
        cursor.execute(f"SET work_mem='{work_mem}';")
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")
        cursor.execute(
            f"SET max_parallel_workers_per_gather={max_parallel_workers_per_gather};")
        cursor.execute(f"SET ivfflat.probes={probes};")

        cursor.execute(
            f"""
            EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)
            SELECT {table_name}.chunk_id, chunks.doi, chunks.text, {table_name}.embedding {operator} %s AS distance 
            FROM {table_name} 
            JOIN chunks ON {table_name}.chunk_id = chunks.id
            ORDER BY embedding {operator} %s DESC 
            LIMIT %s;
            """,
            (query_vector, query_vector, top_k)
        )
        results = cursor.fetchall()
        print(f"Explain results:\n{results}")
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        cursor.close()

        # Define the named tuple
        VectorQueryResult = namedtuple(
            'VectorQueryResult', ['chunk_id', 'doi', 'text', 'similarity'])
        return [VectorQueryResult(*result) for result in results]

    def test_connection(self):
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        # Print a pretty table showing db name, user, host, and port from db_params
        print("="*29 + "CONFIG" + "="*29)
        print(f"{'Database':<16} {'User':<16} {'Host':<32} {'Port':<16}")
        print(
            f"{self.db_params['dbname']:<16} {self.db_params['user']:<16} {self.db_params['host']:<32} {self.db_params['port']:<16}")
        print("="*64)

        # Execute a simple query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"Database version: {db_version}")

        cursor.close()
        conn.close()

    def set_ivf_probes(self, n=None):
        """
        Sets the number of probes for the IVFFlat index. If `n` not specified, it will be set to N**0.25,
        where N is the number of chunks.
        """
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        if n:
            cursor.execute(f"SET ivfflat.probes={n};")
        else:
            # Get the number of chunks
            cursor.execute("SELECT COUNT(*) FROM chunks;")
            n = cursor.fetchone()[0]
            print(f"Setting IVFFlat probes to {n}**0.25 = {int(n**0.25)}")
            cursor.execute(f"SET ivfflat.probes={int(n**0.25)};")

        conn.commit()
        cursor.close()
        conn.close()


def profile_create_vector_table(db, table_name, embedder):
    with cProfile.Profile() as pr:
        db.create_vector_table(name=table_name, dim=768, embedder=embedder)
    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("cumulative").print_stats(40)


def main():
    # NOTE: assumes .env is in the parent directory
    load_dotenv('../.env', override=True)
    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
    print(f"Database parameters: {db_params}")
    db = DatabaseProcessor(db_params)

    args = argument_parser()

    if args.profile:
        from Embedders import get_embedder
        print(
            f"Creating vector table '{'test_table'}' with embedder {'astrobert'} on device {db.device}...")
        embedder = get_embedder(
            model_name="adsabs/astroBERT", device=db.device, normalize=False)
        profile_create_vector_table(db, 'test_table', embedder=embedder)

    if args.add_chunks:
        db.chunk_and_insert_records(args.add_chunks)
        return

    if args.create_vector_table:

        # Extract parameters
        table_name, embedder, normalize, batch_size = args.create_vector_table, args.embedder, args.normalize, args.batch_size

        # Create embedding function and get its dimension
        from Embedders import get_embedder
        print(
            f"Creating vector table '{table_name}' with embedder {embedder} on device {db.device}...")
        embedder = get_embedder(embedder, db.device, normalize)
        dim = embedder(['test']).shape[1]

        db.create_vector_table(
            name=table_name, dim=dim, embedder=embedder, batch_size=batch_size)

        # TODO: add calls to create indexes
        # db.create_index(table_name, 'ivfflat', 'vector_cosine_ops', 1580)
        return

    if args.create_index:
        # Extract parameters
        table_name, index_type, metric = args.table, args.index_type, args.metric
        print(
            f"Creating index on {table_name} with type {index_type} and metric {metric}")
        db.create_index(table_name, index_type, metric)
        return

    if args.test_connection:
        print(f"Connection result:")
        db.test_connection()
        return


if __name__ == '__main__':
    main()
