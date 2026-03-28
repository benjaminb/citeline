"""
ContrastiveDataset: loads HDF5 built by build_dataset.py and applies
a NegativeSelectionConfig to select and weight negatives at train time.

Each __getitem__ returns:
  query      float32 (dim,)
  positive   float32 (dim,)
  negatives  float32 (num_negatives, dim)
  weights    float32 (num_negatives,)   — normalized to sum to num_negatives
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from citeline.nn.contrastive.config import NegativeSelectionConfig


def _compute_weights(
    cfg: NegativeSelectionConfig,
    cosine_sims: np.ndarray,    # (num_negatives,)
    retrieval_ranks: np.ndarray,  # (num_negatives,)
) -> np.ndarray:
    n = len(cosine_sims)
    scheme = cfg.weight_scheme

    if scheme == "uniform":
        w = np.ones(n, dtype=np.float32)

    elif scheme == "bins":
        w = np.ones(n, dtype=np.float32)
        bin_weights = cfg.bin_weights
        num_bins = cfg.num_bins
        # pad/truncate bin_weights to num_bins
        bw = (bin_weights + [bin_weights[-1]] * num_bins)[:num_bins]
        bin_edges = np.linspace(0, n, num_bins + 1, dtype=int)
        for b in range(num_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            w[lo:hi] = bw[b]

    elif scheme == "cosine_sim":
        if cfg.cosine_transform == "softmax":
            logits = cosine_sims / cfg.cosine_temperature
            logits -= logits.max()
            exp = np.exp(logits)
            w = (exp / exp.sum()).astype(np.float32)
            # already sums to 1; will be rescaled below
        else:  # linear
            w = cosine_sims.astype(np.float32)
            w = np.clip(w, 0, None)

    elif scheme == "inv_rank":
        alpha = cfg.inv_rank_alpha
        w = (1.0 / (retrieval_ranks + 1) ** alpha).astype(np.float32)

    else:
        raise ValueError(f"Unknown weight_scheme: {cfg.weight_scheme!r}")

    # Normalize so weights sum to num_negatives (keeps loss scale-invariant)
    total = w.sum()
    if total > 0:
        w = w * (n / total)
    return w


class ContrastiveDataset(Dataset):
    def __init__(self, h5_path: str, cfg: NegativeSelectionConfig):
        self.cfg = cfg
        lo, hi = cfg.rank_range
        with h5py.File(h5_path, "r") as f:
            self.queries = f["queries"][:]                      # (N, dim)
            self.positives = f["positives"][:]                  # (N, dim)
            self.negatives = f["negatives"][:, lo:hi, :]        # (N, hi-lo, dim)
            self.neg_cosine_sims = f["neg_cosine_sims"][:, lo:hi]   # (N, hi-lo)
            self.neg_retrieval_ranks = f["neg_retrieval_ranks"][:, lo:hi]  # (N, hi-lo)

        pool_size = self.negatives.shape[1]
        if cfg.num_negatives > pool_size:
            raise ValueError(
                f"num_negatives={cfg.num_negatives} exceeds pool size {pool_size} "
                f"for rank_range={cfg.rank_range}"
            )

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        pool_size = self.negatives.shape[1]
        n = self.cfg.num_negatives

        # Sample without replacement from the rank_range pool
        chosen = np.random.choice(pool_size, size=n, replace=False)
        chosen.sort()  # preserve rank order for bin-based weighting

        neg_vecs = self.negatives[idx, chosen]          # (n, dim)
        cos_sims = self.neg_cosine_sims[idx, chosen]    # (n,)
        ranks = self.neg_retrieval_ranks[idx, chosen]   # (n,)

        weights = _compute_weights(self.cfg, cos_sims, ranks)  # (n,)

        return (
            torch.from_numpy(self.queries[idx]),
            torch.from_numpy(self.positives[idx]),
            torch.from_numpy(neg_vecs),
            torch.from_numpy(weights),
        )
