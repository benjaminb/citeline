import argparse
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from citeline.nn.config import TrainConfig
from citeline.nn.contrastive_dataset_builder import ContrastiveDatasetBuilder
from citeline.nn.ranking_strategies import RankingStrategy
from citeline.nn.contrastive_datasets import ContrastiveDataset
from citeline.nn.loss_functions import ContrastiveLossFunction
from citeline.nn.loss_schedules import LossSchedule
from citeline.nn.models import Adapter


def build_dataloaders(writer: ContrastiveDatasetBuilder, adapter: Adapter, dataset_cls: ContrastiveDataset, batch_size: int = 32):
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


def run_epoch(model, train_loader, val_loader, loss_fn, optimizer) -> float:
    """
    Returns:
    - train_loss: average loss across all training batches
    - val_loss: average loss across all validation batches
    """
    device = next(model.parameters()).device
    total_train_loss, total_val_loss = 0.0, 0.0
    model.train()
    for query, positives, negatives in train_loader:
        query, pos, neg = [t.to(device) for t in (query, positives, negatives)]
        anchor = model(query)
        loss = loss_fn(anchor, pos, neg, training=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * len(query)

    model.eval()
    with torch.no_grad():
        for query, positives, negatives in val_loader:
            query, pos, neg = [t.to(device) for t in (query, positives, negatives)]
            anchor = model(query)
            loss = loss_fn(anchor, pos, neg, training=False)
            total_val_loss += loss.item() * len(query)
    return total_train_loss / len(train_loader.dataset), total_val_loss / len(val_loader.dataset)


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


# Get --config argument for training config yaml path
def parse_args():
    parser = argparse.ArgumentParser(description="Train a contrastive model with the given config.")
    parser.add_argument("config", type=str, nargs="?", help="Path to the training config YAML file.")
    parser.add_argument("--config", dest="config", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = args.config
    train_config = TrainConfig.from_yaml(config_path)
    print(f"Loaded training config: {train_config}")

    # Get model
    model = Adapter.registry[train_config.model]()
    model = model.to("mps")
    print(f"Initialized model: {model.description}")

    # Ensure checkpoint path exists
    checkpoint_path = Path(train_config.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint will save to: \033[1;34m{checkpoint_path}\033[0m")

    # Build dataset
    strategy = RankingStrategy.registry[train_config.strategy]()

    h5_dataset_writer = ContrastiveDatasetBuilder(
        dataset_dir=train_config.parquet_datadir,
        output_dir=train_config.h5_datadir,
        strategy=strategy,
        adapter=model,
        num_positives=train_config.num_positives,
        num_negatives=train_config.num_negatives,
    )
    dataset_cls = ContrastiveDataset.registry[train_config.dataset_class]
    dataloaders = build_dataloaders(
        writer=h5_dataset_writer, adapter=model, dataset_cls=dataset_cls, batch_size=train_config.batch_size
    )
    epochs = train_config.epochs

    # Build loss schedule and loss function
    loss_schedule = None
    if train_config.loss_schedule:
        total_steps = epochs * len(dataloaders["train"])
        loss_schedule = LossSchedule.registry[train_config.loss_schedule](total_steps=total_steps)
    loss_fn = ContrastiveLossFunction.registry[train_config.loss](loss_schedule=loss_schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    train_losses, val_losses = [], []
    min_val_loss = float("inf")

    for i in range(epochs):
        # Rebuild dataloaders every 10 epochs to refresh the H5 datasets with the current model's embeddings
        if i % 10 == 0 and i > 0:
            print(f"\nEpoch {i+1}: Rebuilding H5 datasets and dataloaders with current model embeddings...")
            # Write H5 datasets (train/val/test)
            dataloaders = build_dataloaders(
                writer=h5_dataset_writer, adapter=model, dataset_cls=dataset_cls, batch_size=train_config.batch_size
            )
        train_loss, val_loss = run_epoch(
            model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Checkpoint
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            # Save the best model checkpoint
            torch.jit.script(model).save(str(checkpoint_path / "best_model_scripted.pt"))
            print(f"New best model saved with val loss: {min_val_loss:.4f}")

        # TODO: implement early stopping
        print(f"Epoch {i+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print(f"Best model saved to: \033[1;34m{checkpoint_path / 'best_model.pth'}\033[0m")
    print(f"TorchScript model saved to: \033[1;34m{checkpoint_path / 'best_model_scripted.pt'}\033[0m")

    _plot_history(
        history=[
            {"epoch": i + 1, "train_loss": tl, "val_loss": vl}
            for i, (tl, vl) in enumerate(zip(train_losses, val_losses))
        ],
        test_loss=val_losses[-1],
        out_path=checkpoint_path / "training_history.png",
    )


if __name__ == "__main__":
    main()
