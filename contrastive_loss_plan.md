# Contrastive Learning Dataset System

## Context

We need a flexible system for training a query-only neural net mapper using InfoNCE/NT-Xent contrastive loss. The net maps query embeddings (from citing sentences) closer to their target document chunk embeddings. The key challenge: supporting multiple negative sampling and weighting strategies without rebuilding datasets for each experiment.

**Core insight**: Build one HDF5 dataset per embedding model that stores ALL top-K negatives with metadata (cosine similarity, retrieval rank). Different experiment configs then select subsets and compute weights at train time from the same file.

---

## File Structure

```
src/citeline/nn/contrastive/
    __init__.py
    build_dataset.py       # Milvus -> HDF5 pipeline
    dataset.py             # PyTorch Dataset (loads HDF5, applies weight config)
    loss.py                # Weighted InfoNCE loss
    model.py               # QueryMapper (query-only MLP)
    train.py               # Training loop with early stopping
    config.py              # Dataclasses + YAML loading
configs/contrastive/
    build.yaml             # Dataset build config
    train_hard_only.yaml   # Example: only hard negatives, uniform weight
    train_hard_easy.yaml   # Example: hard + easy with bin weights
    train_cosine.yaml      # Example: cosine similarity weighting
    train_rank.yaml        # Example: inverse rank weighting
```

---

## HDF5 Schema (one file per split: train/val/test)

```
/queries              float32 (N, dim)       # query embeddings
/positives            float32 (N, dim)       # best positive chunk per query
/negatives            float32 (N, K, dim)    # top-K non-target chunks
/neg_cosine_sims      float32 (N, K)         # cosine sim of each negative to query
/neg_retrieval_ranks  int32   (N, K)         # 0-indexed rank among non-targets
attrs: embedder, collection, dim, K, build_date, dataset_source
```

K ~200 gives enough range to support hard-only, easy-only, binned, and continuous weighting.

---

## Key Components

### 1. `config.py` - Configuration

- `BuildConfig`: dataset_source, embedder, collection, num_negatives (K), milvus_top_k, split fractions
- `NegativeSelectionConfig`: num_negatives (at train time), rank_range [lo, hi), weight_scheme, scheme-specific params
- `TrainConfig`: dataset_dir, model arch (hidden_dims, dropout), negative_selection, temperature, lr, epochs, patience

All loaded from YAML via `from_yaml()` classmethods.

### 2. `build_dataset.py` - Dataset Building Pipeline

Steps:
1. Load `nontrivial_llm.jsonl`, explode on `citation_dois` -> one row per (query, target_doi)
2. Embed all `sent_no_cit` with the configured embedder
3. Split 70/15/15
4. For each split, for each row:
   - **Positive**: `db.select_by_doi(target_doi)` -> get all chunks -> pick most similar to query
   - **Negatives**: `db.search(query_vector, limit=top_k)` -> filter out target DOIs -> keep first K non-targets -> record vector, cosine sim (from Milvus `metric` field), and rank (0-indexed)
5. Write to HDF5

Uses existing: `Embedder.create()`, `MilvusDB.search()`, `MilvusDB.select_by_doi()`

### 3. `dataset.py` - PyTorch Dataset

`ContrastiveDataset.__getitem__(idx)`:
1. Slice negatives to `rank_range` (e.g., [0, 50] for hard-only, [150, 200] for easy-only)
2. Sample `num_negatives` from that slice
3. `compute_weights()` based on scheme:
   - **uniform**: all 1.0
   - **bins**: divide selected negatives into k bins by rank, assign per-bin weight
   - **cosine_sim**: weight = cosine_sim (optionally with softmax temperature)
   - **inv_rank**: weight = 1/(rank+1)^alpha
4. Normalize weights to sum to `num_negatives` (keeps loss scale-invariant)
5. Return `(query, positive, negatives, weights)` as tensors

### 4. `loss.py` - Weighted InfoNCE

```
L = -sim(q, p+)/tau + log( exp(sim(q, p+)/tau) + sum_j w_j * exp(sim(q, n_j)/tau) )
```

When all w_j = 1.0, this is standard InfoNCE. Higher weights on hard negatives amplify their gradient signal.

Implementation uses `log(w_j) + sim_neg_j` trick with `logsumexp` for numerical stability.

### 5. `model.py` - QueryMapper

MLP with configurable hidden dims, GELU activations, dropout, L2-normalized output. Only query vectors pass through the model; target/negative vectors remain fixed.

### 6. `train.py` - Training Loop

Standard PyTorch loop: forward mapped queries through model, compute weighted InfoNCE against fixed positives/negatives, AdamW optimizer, early stopping on validation loss.

---

## Experiment Variations (all use the same HDF5 dataset)

| Experiment | rank_range | weight_scheme | Key params |
|---|---|---|---|
| Hard negatives only | [0, 50] | uniform | -- |
| Hard + easy | [0, 200] | bins | num_bins=2, weights=[2.0, 1.0] |
| k-bin graduated | [0, 200] | bins | num_bins=4, weights=[4, 2, 1, 0.5] |
| Cosine similarity | [0, 200] | cosine_sim | transform=linear |
| Retrieval rank | [0, 200] | inv_rank | alpha=1.0 |

---

## Verification

1. **Build**: Run `python -m citeline.nn.contrastive.build_dataset configs/contrastive/build.yaml` -> produces `train.h5`, `val.h5`, `test.h5`
2. **Inspect**: Load HDF5, verify shapes, spot-check that negatives' cosine sims decrease with rank
3. **Train**: Run `python -m citeline.nn.contrastive.train configs/contrastive/train_hard_only.yaml` -> verify loss decreases, model checkpoint saved
4. **Compare**: Run multiple train configs on same dataset, compare val loss curves
