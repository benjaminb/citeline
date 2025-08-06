from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import os
from dotenv import load_dotenv

def create_contributions_collection():
    """
    Create a Milvus collection equivalent to the PostgreSQL contributions table
    """
    # Load environment variables for Milvus connection
    load_dotenv("../.env")
    
    # Connect to Milvus
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
        user=os.getenv("MILVUS_USER", ""),
        password=os.getenv("MILVUS_PASSWORD", "")
    )
    
    # Define fields schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="bge", dtype=DataType.FLOAT_VECTOR, dim=1024),  # BGE embeddings (current)
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="pubdate", dtype=DataType.VARCHAR, max_length=10),  # Store as YYYY-MM-DD string
        FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=255)
        # Future vector fields can be added here or dynamically:
        # FieldSchema(name="e5", dtype=DataType.FLOAT_VECTOR, dim=768),  # E5 embeddings
        # FieldSchema(name="nomic", dtype=DataType.FLOAT_VECTOR, dim=768),  # Nomic embeddings
        # FieldSchema(name="cohere", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Cohere embeddings
    ]
    
    # Create collection schema
    schema = CollectionSchema(
        fields=fields,
        description="Contributions collection migrated from PostgreSQL"
    )
    
    # Create collection
    collection_name = "contributions"
    if Collection.exists(collection_name):
        print(f"Collection '{collection_name}' already exists")
        collection = Collection(collection_name)
    else:
        collection = Collection(name=collection_name, schema=schema)
        print(f"Created collection '{collection_name}'")
    
    # Create index on vector field for similarity search
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    collection.create_index(field_name="bge", index_params=index_params)
    print("Created index on bge field")
    
    # Load collection into memory
    collection.load()
    print("Collection loaded into memory")
    
    return collection


def create_chunks_collection():
    """
    Create a Milvus collection equivalent to the PostgreSQL chunks table
    """
    # Connect to Milvus (assuming connection is already established)
    
    # Define fields schema for chunks
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="bge", dtype=DataType.FLOAT_VECTOR, dim=1024),  # BGE embeddings (current)
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="pubdate", dtype=DataType.VARCHAR, max_length=10),  # Store as YYYY-MM-DD string
        FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=255)
        # Future vector fields for different embedders:
        # FieldSchema(name="e5", dtype=DataType.FLOAT_VECTOR, dim=768),
        # FieldSchema(name="nomic", dtype=DataType.FLOAT_VECTOR, dim=768),
        # FieldSchema(name="cohere", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    
    # Create collection schema
    schema = CollectionSchema(
        fields=fields,
        description="Chunks collection migrated from PostgreSQL"
    )
    
    # Create collection
    collection_name = "chunks"
    if Collection.exists(collection_name):
        print(f"Collection '{collection_name}' already exists")
        collection = Collection(collection_name)
    else:
        collection = Collection(name=collection_name, schema=schema)
        print(f"Created collection '{collection_name}'")
    
    # Create index on BGE vector field
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    collection.create_index(field_name="bge", index_params=index_params)
    print("Created index on bge field for chunks")
    
    # Load collection into memory
    collection.load()
    print("Chunks collection loaded into memory")
    
    return collection


def add_vector_field_to_collection(collection_name: str, field_name: str, dim: int, 
                                   index_type: str = "IVF_FLAT", metric_type: str = "COSINE"):
    """
    Add a new vector field to an existing Milvus collection
    
    Args:
        collection_name: Name of the collection to modify
        field_name: Name of the new vector field (e.g., 'e5', 'nomic', 'cohere')
        dim: Dimension of the vector field
        index_type: Type of index to create (default: IVF_FLAT)
        metric_type: Distance metric to use (default: COSINE)
    """
    # Note: As of current Milvus versions, you cannot add fields to existing collections
    # You would need to:
    # 1. Create a new collection with the additional field
    # 2. Migrate data from the old collection
    # 3. Drop the old collection and rename the new one
    
    print(f"Warning: Adding fields to existing collections requires data migration.")
    print(f"Consider planning all vector fields upfront or implement a migration strategy.")
    
    # This is a placeholder for the migration logic
    # In practice, you'd need to:
    # 1. Export all data from the existing collection
    # 2. Create a new collection with the additional field
    # 3. Re-insert all data with the new field (initially empty/null)
    # 4. Populate the new field with embeddings
    pass


if __name__ == "__main__":
    # Create contributions collection
    contributions_collection = create_contributions_collection()
    print(f"Contributions collection info: {contributions_collection.describe()}")
    
    # Create chunks collection  
    chunks_collection = create_chunks_collection()
    print(f"Chunks collection info: {chunks_collection.describe()}")
