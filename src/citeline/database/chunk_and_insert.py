import argparse
import pandas as pd
from semantic_text_splitter import TextSplitter
from citeline.database.milvusdb import MilvusDB


"""

1. open the jsonl file expected to have the following keys:
- text
- id
- pubdate (YYYYMMDD)
    - optional: citation_count
2. Use TextSplitter to split the text


"""
splitter = TextSplitter(capacity=1500, overlap=150)


def argument_parser():
    parser = argparse.ArgumentParser(description="Chunk and insert documents into MilvusDB")
    parser.add_argument(
        "--infile",
        type=str,
        help="Path to the input .jsonl file (unchunked reference docs with text, doc_id, and pubdate fields)",
    )
    parser.add_argument("--outfile", type=str, help="Path to the output .jsonl file (chunked reference docs)")
    parser.add_argument("--collection", type=str, required=True, help="Name for the MilvusDB collection")
    parser.add_argument("--embedder", type=str, required=True, help="Name of the embedding model to use")
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Whether to normalize embeddings before inserting into MilvusDB",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding generation and insertion")

    args = parser.parse_args()
    return args


def load_jsonl(filepath: str):
    """Loads a .jsonl file and checks preconditions:"""
    df = pd.read_json(filepath, lines=True)
    required_columns = {"text", "doc_id", "pubdate"}
    columns_present = set(df.columns)
    assert required_columns.issubset(columns_present), f"File is missing columns: {required_columns - columns_present}"
    return df


def create_chunk_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the global TextSplitter to chunk the 'text' field of each row in the dataframe.

    Precondition: df has columns: text, doc_id, pubdate, optional citation_count
    Postcondition: returns a dataframe with these columns: renamed doc_id to doi
    """
    df_has_citation_counts = "citation_count" in df.columns
    chunked_records = []
    for row in df.itertuples():
        text, doc_id, pubdate = row.text, row.doc_id, row.pubdate
        citation_count = -1
        if df_has_citation_counts:
            citation_count = row.citation_count
        chunks = splitter.chunks(text)
        chunks = [chunk.strip().replace("\x00", "") for chunk in chunks if chunk.strip()]
        for chunk in chunks:
            record = {
                "doi": doc_id,
                "text": chunk,
                "pubdate": pubdate,
                "citation_count": citation_count,
            }
            chunked_records.append(record)
    df_chunked = pd.DataFrame(chunked_records)
    df.rename(columns={"doc_id": "doi"}, inplace=True)
    return df_chunked


def save_chunked_jsonl(df: pd.DataFrame, filepath: str) -> None:
    df.to_json(filepath, lines=True, orient="records")


def main():
    args = argument_parser()
    print(f"Args: {args}")
    reference_path = args.infile
    save_chunked_path = args.outfile
    collection_name = args.collection
    embedder_name = args.embedder
    normalize = args.normalize
    batch_size = args.batch_size

    # Load and chunk reference docs
    df = load_jsonl(reference_path)
    chunked_df = create_chunk_df(df)
    save_chunked_jsonl(chunked_df, save_chunked_path)

    # Insert into milvus
    db = MilvusDB()
    db.create_vector_collection_pd(
        name=collection_name, data=chunked_df, embedder_name=embedder_name, normalize=normalize, batch_size=batch_size
    )


if __name__ == "__main__":
    main()
