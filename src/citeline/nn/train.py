import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from citeline.nn.config import H5DatasetWriterConfig, TrainConfig
from citeline.nn.build_h5_datasets import H5DatasetWriter
from citeline.nn.contrastive_datasets import ContrastiveDataset
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


def run_epoch(model, loader, loss_fn, optimizer, temperature, device, train_mode_on: bool) -> float:
    # Set model training mode
    model.train(train_mode_on)
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
    model_cls = Adapter.registry[train_config.model]
    model = model_cls()
    print(f"Initialized model: {model.description}")

    # Build dataset
    dataset_config = "src/citeline/nn/configs/create_test_h5.yaml"
    h5_dataset_writer = H5DatasetWriter.from_config(dataset_config)
    dataset_cls = ContrastiveDataset.registry[train_config.dataset_class]

    # Write H5 datasets (train/val/test)
    dataloaders = build_dataloaders(
        writer=h5_dataset_writer,
        adapter=model,
        dataset_cls=dataset_cls,
        batch_size=train_config.batch_size
    )
    print("Dataloaders built:")
    print(dataloaders)

    epochs = train_config.epochs
    for i in range(epochs):
        train_loss = run_epoch(model, dataloaders["train"], None, None, train_config.temperature, "cpu", True)
        val_loss = run_epoch(model, dataloaders["val"], None, None, train_config.temperature, "cpu", False)
        print(f"Epoch {i+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
