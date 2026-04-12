from abc import ABC, abstractmethod
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ContrastiveDataset(ABC, Dataset):
    registry = {}

    def __init__(self, h5_path: str):
        # Load h5 dataset
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["queries"].shape[0]
            self.queries = torch.tensor(f["queries"][:], dtype=torch.float32)
            self.labels = torch.tensor(f["labels"][:], dtype=torch.float32)

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.registry[cls.__name__] = cls

    def __len__(self):
        return self.length
    
    # Subclasses only need to implement getitem
    @abstractmethod
    def __getitem__(self, idx): ...

class BasicTripletDataset(ContrastiveDataset):
    """
    Loads one (anchor, positive, negative) per sample
    positive: picks the first positive (most similar) for each anchor. If they query has
      multiple targets, it picks one of the targets at random, then picks its most similar positive.
    negative: picks the first negative (most similar) for each anchor.
    """
    def __init__(self, h5_path: str):
        # Load h5 dataset
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["queries"].shape[0]
            self.anchors = torch.tensor(f["queries"][:], dtype=torch.float32)
            self.num_targets = f["num_targets"][:]
            self.positives = torch.tensor(f["positives"][:], dtype=torch.float32)
            self.negatives = torch.tensor(f["negatives"][:], dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        # Pick one positive of all the available positives
        positive = self.positives[idx, np.random.randint(self.num_targets[idx])]
        return anchor, positive, self.negatives[idx][0]



# def build_dataloader(
#     batch_size: int,
#     shuffle: bool = True,
#     num_workers: int = 0,
# ) -> DataLoader:
#     """
#     Convenience factory. Returns a DataLoader whose batches are tuples of:
#         query      (batch, dim)
#         positives  (batch, num_positives, dim)
#         negatives  (batch, num_negatives, dim)

#     Args:
#         h5_path: path to HDF5 dataset
#         batch_size: samples per batch
#         negative_keys: H5 keys to load as negatives; concatenated if multiple.
#             Defaults to ["negatives"].
#         num_positives: subsample positives per sample. None = use all.
#         num_negatives: subsample negatives per sample (after concat). None = use all.
#         shuffle: whether to shuffle (True for train, False for val/test)
#         num_workers: DataLoader worker processes (keep 0 when h5py file handle is shared)
#     """

#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
