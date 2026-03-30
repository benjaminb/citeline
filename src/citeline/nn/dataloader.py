import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BasicTripletDataset(Dataset):
    """
    Loads one (anchor, positive, negative) per sample
    """
    def __init__(self, h5_path: str):
        # Load h5 dataset
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["queries"].shape[0]
            self.anchors = torch.tensor(f["queries"][:], dtype=torch.float32)
            self.positives = torch.tensor(f["positives"][:], dtype=torch.float32)
            self.negatives = torch.tensor(f["negatives"][:], dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        
        return self.anchors[idx], self.positives[idx], self.negatives[idx]



def build_dataloader(
    h5_path: str,
    batch_size: int,
    negative_keys: list[str] = None,
    num_positives: int = None,
    num_negatives: int = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience factory. Returns a DataLoader whose batches are tuples of:
        query      (batch, dim)
        positives  (batch, num_positives, dim)
        negatives  (batch, num_negatives, dim)

    Args:
        h5_path: path to HDF5 dataset
        batch_size: samples per batch
        negative_keys: H5 keys to load as negatives; concatenated if multiple.
            Defaults to ["negatives"].
        num_positives: subsample positives per sample. None = use all.
        num_negatives: subsample negatives per sample (after concat). None = use all.
        shuffle: whether to shuffle (True for train, False for val/test)
        num_workers: DataLoader worker processes (keep 0 when h5py file handle is shared)
    """
    dataset = EmbeddingTripletDataset(
        h5_path=h5_path,
        negative_keys=negative_keys,
        num_positives=num_positives,
        num_negatives=num_negatives,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
