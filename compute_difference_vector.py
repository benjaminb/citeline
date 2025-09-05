import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from embedders import Embedder
from database.milvusdb import MilvusDB

tqdm.pandas()


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create(model_name="Qwen/Qwen3-Embedding-0.6B", device=device, normalize=True, for_queries=True)
    db = MilvusDB()
    db.list_collections()


if __name__ == "__main__":
    main()
