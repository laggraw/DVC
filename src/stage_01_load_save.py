from pandas.core.indexes.base import Index
from src.utils.all_utils import read_yaml, create_directory
import argparse
import pandas as pd
import os


def get_data(config_path):
    config = read_yaml(config_path)

    remote_data_path = config["data_source"]
    df = pd.read_csv(remote_data_path, sep = ";")

    # save the dataset in local directory
    # create directory path : artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"] 
    raw_local_dir = config["artifacts"]["raw_local_dir"] 
    raw_local_file = config["artifacts"]["raw_local_file"]

    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)
    create_directory(dirs = [raw_local_dir_path])
    raw_local_file_path = os.path.join(raw_local_dir_path, raw_local_file)
    df.to_csv( raw_local_file_path, sep=",", index =False)
    # print(raw_local_file_path)
    # print(df.head(20))

if __name__ =='__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_arg = args.parse_args()

    print(parsed_arg.config)

    get_data(config_path = parsed_arg.config)