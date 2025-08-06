import psycopg
import numpy as np
from pymilvus import connections, Collection
from tqdm import tqdm
import os
from dotenv import load_dotenv

def migrate_contributions_table(batch_size=1000):
    """
    Migrate data from PostgreSQL contributions table to Milvus collection
    """
    load_dotenv("../.env")
    
    # Connect to PostgreSQL
    pg_params = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }
    
    # Connect to Milvus
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
        user=os.getenv("MILVUS_USER", ""),
        password=os.getenv("MILVUS_PASSWORD", "")
    )
    
    # Get Milvus collection
    collection = Collection("contributions")
    
    with psycopg.connect(**pg_params) as pg_conn:
        with pg_conn.cursor() as cursor:
            # Get total count for progress bar
            cursor.execute("SELECT COUNT(*) FROM contributions")
            total_rows = cursor.fetchone()[0]
            print(f"Total rows to migrate: {total_rows}")
            
            # Fetch and insert data in batches
            cursor.execute("SELECT id, embedding, text, pubdate, doi FROM contributions ORDER BY id")
            
            batch_data = {
                "id": [],
                "embedding": [],
                "text": [],
                "pubdate": [],
                "doi": []
            }
            
            with tqdm(total=total_rows, desc="Migrating data") as pbar:
                for row in cursor:
                    id_val, embedding_val, text_val, pubdate_val, doi_val = row
                    
                    # Convert PostgreSQL array to list for Milvus
                    embedding_list = embedding_val.tolist() if isinstance(embedding_val, np.ndarray) else list(embedding_val)
                    
                    batch_data["id"].append(id_val)
                    batch_data["embedding"].append(embedding_list)
                    batch_data["text"].append(text_val)
                    batch_data["pubdate"].append(pubdate_val.strftime("%Y-%m-%d") if pubdate_val else "")
                    batch_data["doi"].append(doi_val)
                    
                    # Insert batch when it reaches batch_size
                    if len(batch_data["id"]) >= batch_size:
                        collection.insert(batch_data)
                        pbar.update(len(batch_data["id"]))
                        
                        # Clear batch
                        for key in batch_data:
                            batch_data[key] = []
                
                # Insert remaining data
                if batch_data["id"]:
                    collection.insert(batch_data)
                    pbar.update(len(batch_data["id"]))
    
    # Flush to ensure data is persisted
    collection.flush()
    print("Migration completed successfully!")

if __name__ == "__main__":
    migrate_contributions_table()
```
