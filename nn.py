from torch.utils.data import DataLoader
from torchgen import model

from citeline.nn.build_raw_datasets import build_dataset
from citeline.nn.build_h5_datasets import H5DatasetWriter
from citeline.nn.contrastive_datasets import ContrastiveDataset
from citeline.nn.models import Adapter

def build_dataloaders(
        writer: H5DatasetWriter, 
        dataset_cls: ContrastiveDataset,
        batch_size: int = 32
    ):
    h5_paths = writer.write_h5()
    dataloaders = {}
    for split in ("train", "val", "test"):
        h5_path = h5_paths[split]
        dataset = dataset_cls(h5_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))
        dataloaders[split] = dataloader
        # NOTE: any advantage to yielding instead of returning?
    return dataloaders


def main():
    # config_path = "src/citeline/nn/configs/test_dataset_config.yaml"
    # build_dataset(config_path)

    h5_build_config_path = "src/citeline/nn/configs/create_test_h5.yaml"
    writer = H5DatasetWriter.from_config(h5_build_config_path)
    h5_paths = writer.write_h5()
    print(f"H5 datasets written to: {h5_paths}")

    # Instantiate model
    model_cls = Adapter.registry.get("BaselineMLPEmbeddingMapper")
    if model_cls is None:
        raise ValueError(
            f"Model 'BaselineMLPEmbeddingMapper' not found in registry. Available models: {list(Adapter.registry.keys())}"
        )
    model = model_cls()
    print(f"Initialized model: {model.description}")

    # Build dataloaders
    dataloaders = build_dataloaders(
        writer=writer,
        adapater=model,
        dataset_cls=ContrastiveDataset.registry["BasicTripletDataset"],
        batch_size=32
    )
    print("Dataloaders built:")
    print(dataloaders)


if __name__ == "__main__":
    main()
