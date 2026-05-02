import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader

from citeline.nn.config import TrainConfig
from citeline.nn.contrastive_dataset_writer import ContrastiveDatasetWriter
from citeline.nn.ranking_strategies import RankingStrategy
from citeline.nn.contrastive_datasets import ContrastiveDataset
from citeline.nn.loss_functions import ContrastiveLossFunction
from citeline.nn.loss_schedules import LossSchedule
from citeline.nn.models import Adapter

REBUILD_INTERVAL = 40  # epochs

def build_dataloaders(writer: ContrastiveDatasetWriter, adapter: Adapter, dataset_cls: ContrastiveDataset, batch_size: int = 32):
    # Slot in the current adapter, write out current search results
    writer.adapter = adapter
    h5_paths = writer.write_h5()

    # Build dataloaders for train/val/test
    dataloaders = {}
    for split in ("train", "val", "test"):
        h5_path = h5_paths[split]
        dataset = dataset_cls(h5_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
        dataloaders[split] = dataloader
        # NOTE: any advantage to yielding instead of returning?
    return dataloaders


def run_epoch(model, train_loader, val_loader, loss_fn, optimizer):
    """
    Returns:
    - train_loss: average loss across all training batches
    - val_loss: average loss across all validation batches
    - train_pos_sim: mean cosine similarity between anchor and positive (train)
    - train_neg_sim: mean cosine similarity between anchor and negative (train)
    """
    device = next(model.parameters()).device
    total_train_loss, total_val_loss = 0.0, 0.0
    total_pos_sim, total_neg_sim = 0.0, 0.0
    model.train()
    for query, positives, negatives in train_loader:
        query, pos, neg = [t.to(device) for t in (query, positives, negatives)]
        anchor = model(query)
        loss = loss_fn(anchor, pos, neg, training=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * len(query)
        total_pos_sim += F.cosine_similarity(anchor.detach(), pos).sum().item()
        total_neg_sim += F.cosine_similarity(anchor.detach(), neg).sum().item()

    total_val_pos_sim, total_val_neg_sim = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for query, positives, negatives in val_loader:
            query, pos, neg = [t.to(device) for t in (query, positives, negatives)]
            anchor = model(query)
            loss = loss_fn(anchor, pos, neg, training=False)
            total_val_loss += loss.item() * len(query)
            total_val_pos_sim += F.cosine_similarity(anchor, pos).sum().item()
            total_val_neg_sim += F.cosine_similarity(anchor, neg).sum().item()

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    return (
        total_train_loss / n_train,
        total_val_loss / n_val,
        total_pos_sim / n_train,
        total_neg_sim / n_train,
        total_val_pos_sim / n_val,
        total_val_neg_sim / n_val,
    )


def _plot_history(history: list[dict], test_loss: float, out_path: Path) -> None:
    epochs = [e["epoch"] for e in history]
    train_losses = [e["train_loss"] for e in history]
    val_losses = [e["val_loss"] for e in history]
    train_margins = [e["train_margin"] for e in history]
    val_margins = [e["val_margin"] for e in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.axhline(test_loss, color="tab:red", linestyle="--", label=f"test ({test_loss:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Contrastive Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    margin_path = out_path.parent / "margin_history.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_margins, label="train margin")
    ax.plot(epochs, val_margins, label="val margin")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Margin (pos − neg)")
    ax.set_title("Contrastive Training — Margin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(margin_path, dpi=150)
    plt.close(fig)


def run_training(config: TrainConfig, checkpoint_dir: str | None = None) -> Path:
    """
    Train an adapter model given a TrainConfig.

    Args:
        config: TrainConfig instance with all training parameters.
        checkpoint_dir: Directory to save checkpoints. If None, uses config.checkpoint_path.

    Returns:
        Path to the best_model.pt checkpoint.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else Path(config.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint will save to: \033[1;34m{checkpoint_path}\033[0m")

    model = Adapter.registry[config.model]()
    model = model.to(device)
    print(f"Initialized model: {model.description}")

    strategy = RankingStrategy.registry[config.strategy]()
    h5_dataset_writer = ContrastiveDatasetWriter(
        dataset_dir=config.parquet_datadir,
        output_dir=config.h5_datadir,
        strategy=strategy,
        adapter=model,
        num_positives=config.num_positives,
        num_negatives=config.num_negatives,
    )
    dataset_cls = ContrastiveDataset.registry[config.dataset_class]
    dataloaders = build_dataloaders(
        writer=h5_dataset_writer, adapter=model, dataset_cls=dataset_cls, batch_size=config.batch_size
    )
    epochs = config.epochs

    loss_schedule = None
    if config.loss_schedule:
        total_steps = epochs * len(dataloaders["train"])
        loss_schedule = LossSchedule.registry[config.loss_schedule](total_steps=total_steps)
        print(f"Total training steps: \033[1;34m{total_steps}\033[0m")
    loss_fn = ContrastiveLossFunction.registry[config.loss](loss_schedule=loss_schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_losses, val_losses, train_margins, val_margins = [], [], [], []
    best_train_margin = float("-inf")
    best_val_margin = float("-inf")
    rebuild_counter = 0

    for i in range(epochs):
        train_loss, val_loss, pos_sim, neg_sim, val_pos_sim, val_neg_sim = run_epoch(
            model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_margins.append(pos_sim - neg_sim)
        val_margins.append(val_pos_sim - val_neg_sim)

        # Track best train margin for potential dataset rebuild
        if pos_sim - neg_sim > best_train_margin:
            best_train_margin = pos_sim - neg_sim
            rebuild_counter = 0  # reset counter if we see improvement
        else:
            rebuild_counter += 1
            if rebuild_counter >= config.rebuild_patience:
                print(f"\nEpoch {i+1}: Rebuilding H5 datasets and dataloaders with current model embeddings...")
                dataloaders = build_dataloaders(
                    writer=h5_dataset_writer, adapter=model, dataset_cls=dataset_cls, batch_size=config.batch_size
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                rebuild_counter = 0  # reset counter after rebuild
                best_train_margin = float("-inf")  # reset best margin after rebuild

        if val_pos_sim - val_neg_sim > best_val_margin:
            best_val_margin = val_pos_sim - val_neg_sim
            torch.jit.script(model).save(str(checkpoint_path / "best_model.pt"))
            print(f"New best model saved with val margin {best_val_margin:.4f} (pos: {val_pos_sim:.4f}, neg: {val_neg_sim:.4f})")

        print(f"Epoch {i+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | train margin: {pos_sim - neg_sim:.4f} | val margin: {val_pos_sim - val_neg_sim:.4f}")

    torch.jit.script(model).save(str(checkpoint_path / "final_model.pt"))
    print(f"Final model saved to: \033[1;34m{checkpoint_path / 'final_model.pt'}\033[0m")
    print(f"Best model saved to: \033[1;34m{checkpoint_path / 'best_model.pt'}\033[0m")

    _plot_history(
        history=[
            {"epoch": i + 1, "train_loss": tl, "val_loss": vl, "train_margin": tm, "val_margin": vm}
            for i, (tl, vl, tm, vm) in enumerate(zip(train_losses, val_losses, train_margins, val_margins))
        ],
        test_loss=val_losses[-1],
        out_path=checkpoint_path / "loss_history.png",
    )

    return checkpoint_path / "best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a contrastive model with the given config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config YAML file.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_config = TrainConfig.from_yaml(args.config)
    print(f"Loaded training config: {train_config}")
    run_training(train_config)


if __name__ == "__main__":
    main()
