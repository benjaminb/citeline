from citeline.nn.build_raw_datasets import build_dataset
from citeline.nn.build_nn_datasets import H5DatasetWriter

def main():
    # config_path = "src/citeline/nn/configs/test_dataset_config.yaml"
    # build_dataset(config_path)
    
    h5_build_config_path = "src/citeline/nn/configs/create_test_h5.yaml"
    h5_writer = H5DatasetWriter.from_config(h5_build_config_path)
    h5_writer.write_h5(output_path="data/dataset/nn_datasets/testrun_train_dataset.h5")

if __name__ == "__main__":
    main()
