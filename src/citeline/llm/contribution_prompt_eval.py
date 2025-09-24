import pandas as pd
import torch
from tqdm import tqdm
from citeline.database.milvusdb import MilvusDB
from citeline.embedders import Embedder
from citeline.query_expander import get_expander

tqdm.pandas()

EMBEDDER_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
START_COLLECTION = "qwen06_contributions"

class PromptIteration:
    def __init__(
        self,
        dataset_path: str,
        num_samples: int,
        db_collection: str,
        num_hard_examples: int = 2,
        random_state: int = 42,
        embedder_name: str = EMBEDDER_NAME,
        query_expansion: str = "add_prev_3",
        device: str = DEVICE,
    ):
        self.embedder = Embedder.create(model_name=embedder_name, device=device, normalize=True)
        self.expander = get_expander(query_expansion)

        self.db = MilvusDB()
        self.db.client.load_collection(db_collection)


    def _get_hard_examples(example: pd.Serices, n: int=2) -> tuple[list[str], list[float]]:
        pass