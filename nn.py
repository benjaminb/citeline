from src.citeline.nn.bb_dataset_builder import build_dataset


def main():
    config_path = "src/citeline/nn/configs/test_dataset_config.yaml"
    build_dataset(config_path)


if __name__ == "__main__":
    main()
