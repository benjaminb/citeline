import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from citeline.nn.config import H5DatasetWriterConfig, TrainConfig
from citeline.nn.build_h5_datasets import H5DatasetWriter, MultiSimilarityStrategy
from citeline.nn.contrastive_datasets import ContrastiveDataset
from citeline.nn.loss_functions import ContrastiveLossFunction
from citeline.nn.models import Adapter


def build_dataloaders(writer: H5DatasetWriter, adapter: Adapter, dataset_cls: ContrastiveDataset, batch_size: int = 32):
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


def run_epoch(model, loader, loss_fn, optimizer, temperature, train_mode_on: bool) -> float:
    # Set model training mode
    model.train(train_mode_on)
    device = next(model.parameters()).device
    total_loss = 0.0
    with torch.set_grad_enabled(train_mode_on):
        for query, positives, negatives in loader:
            query = query.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            anchor = model(query)
            loss = loss_fn(anchor, positives, negatives)

            if train_mode_on:
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


def main():
    config_path = "src/citeline/nn/configs/testrun_train_config.yaml"
    train_config = TrainConfig.from_yaml(config_path)
    print(f"Loaded training config: {train_config}")

    # Get model
    model = Adapter.registry[train_config.model]()
    model = model.to("mps")
    print(f"Initialized model: {model.description}")
    checkpoint_path = Path(train_config.checkpoint_path)

    # Get loss & optimizers
    loss_fn = ContrastiveLossFunction.registry[train_config.loss]()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    # Build dataset
    strategy = MultiSimilarityStrategy.registry[train_config.strategy]()

    h5_dataset_writer = H5DatasetWriter(
        dataset_dir=train_config.parquet_datadir,
        output_dir=train_config.h5_datadir,
        strategy=strategy,
        adapter=model,
        num_positives=train_config.num_positives,
        num_negatives=train_config.num_negatives,
    )
    dataset_cls = ContrastiveDataset.registry[train_config.dataset_class]

    # Write H5 datasets (train/val/test)
    dataloaders = build_dataloaders(
        writer=h5_dataset_writer, adapter=model, dataset_cls=dataset_cls, batch_size=train_config.batch_size
    )
    print("Dataloaders built:")
    print(dataloaders)

    epochs = train_config.epochs
    train_losses, val_losses = [], []

    for i in range(epochs):
        train_loss = run_epoch(
            model,
            dataloaders["train"],
            loss_fn=loss_fn,
            optimizer=optimizer,
            temperature=train_config.temperature,
            train_mode_on=True,
        )
        train_losses.append(train_loss)

        val_loss = run_epoch(
            model,
            dataloaders["val"],
            loss_fn=loss_fn,
            optimizer=optimizer,
            temperature=train_config.temperature,
            train_mode_on=False,
        )
        val_losses.append(val_loss)
        # TODO: implement checkpointing
        # TODO: implement early stopping
        print(f"Epoch {i+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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
