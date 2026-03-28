"""
Training loop for the QueryMapper.

Usage:
  python -m citeline.nn.contrastive.train configs/contrastive/train_hard_only.yaml
"""

import os
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from citeline.nn.contrastive.config import TrainConfig
from citeline.nn.contrastive.dataset import ContrastiveDataset
from citeline.nn.contrastive.loss import weighted_infonce
from citeline.nn.contrastive.model import build_model


def get_dim(dataset_dir: str) -> int:
    import h5py
    with h5py.File(os.path.join(dataset_dir, "train.h5"), "r") as f:
        return f["queries"].shape[1]


def run_epoch(model, loader, optimizer, temperature, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for query, positive, negatives, weights in loader:
            query = query.to(device)
            positive = positive.to(device)
            negatives = negatives.to(device)
            weights = weights.to(device)

            mapped_query = model(query)
            loss = weighted_infonce(mapped_query, positive, negatives, weights, temperature)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(query)

    return total_loss / len(loader.dataset)


def _plot_history(history: list[dict], test_loss: float, out_path: Path) -> None:
    epochs = [e["epoch"] for e in history]
    train_losses = [e["train_loss"] for e in history]
    val_losses = [e["val_loss"] for e in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.axhline(test_loss, color="tab:red", linestyle="--", label=f"test ({test_loss:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Contrastive Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(config_path: str) -> None:
    cfg = TrainConfig.from_yaml(config_path)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {config_path}")

    dim = get_dim(cfg.dataset_dir)
    model = build_model(dim, cfg.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model.arch}  hidden={cfg.model.hidden_dims}  params={param_count:,}")

    train_ds = ContrastiveDataset(os.path.join(cfg.dataset_dir, "train.h5"), cfg.negative_selection)
    val_ds = ContrastiveDataset(os.path.join(cfg.dataset_dir, "val.h5"), cfg.negative_selection)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, cfg.temperature, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, cfg.temperature, device, train=False)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, cfg.checkpoint_path)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(f"Epoch {epoch:3d}/{cfg.epochs}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epochs_without_improvement >= cfg.patience:
            print(f"Early stopping: no improvement for {cfg.patience} epochs.")
            break

    # Evaluate best checkpoint on test set
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_ds = ContrastiveDataset(os.path.join(cfg.dataset_dir, "test.h5"), cfg.negative_selection)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loss = run_epoch(model, test_loader, None, cfg.temperature, device, train=False)
    print(f"\nTest loss (best checkpoint): {test_loss:.4f}")

    # Save training history alongside checkpoint
    history_path = Path(cfg.checkpoint_path).with_suffix(".json")
    for entry in history:
        entry["test_loss"] = None
    history[-1]["test_loss"] = test_loss  # record final test loss on last entry
    with open(history_path, "w") as f:
        json.dump({"test_loss": test_loss, "history": history}, f, indent=2)

    plot_path = Path(cfg.checkpoint_path).with_suffix(".png")
    _plot_history(history, test_loss, plot_path)

    print(f"Best val loss:  {best_val_loss:.4f}")
    print(f"Checkpoint:     {cfg.checkpoint_path}")
    print(f"History:        {history_path}")
    print(f"Loss plot:      {plot_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m citeline.nn.contrastive.train <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
