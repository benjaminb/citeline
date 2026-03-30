from citeline.nn.build_raw_datasets import build_dataset
from citeline.nn.build_nn_datasets import main as build_nn_datasets

def main():
    config_path = "src/citeline/nn/configs/test_dataset_config.yaml"
    # build_dataset(config_path)
    build_nn_datasets()

if __name__ == "__main__":
    main()
