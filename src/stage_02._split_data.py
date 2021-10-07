from pandas.core.indexes.base import Index
from src.utils.all_utils import read_yaml, create_directory
import argparse
import pandas as pd
import os


def split_save(config_path):
    config = read_yaml(config_path)

    remote_data_path = config["data_source"]
    df = pd.read_csv(remote_data_path, sep = ";")

    # save the dataset in local directory
    # create directory path : artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"] 
    raw_local_dir = config["artifacts"]["raw_local_dir"] 
    raw_local_file = config["artifacts"]["raw_local_file"]

    raw_local_dir_file = os.path.join(artifacts_dir, raw_local_dir,raw_local_file)
    df = pd.read_csv(raw_local_dir_file)
    print(df.head())

if __name__ =='__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_arg = args.parse_args()

    print(parsed_arg.config)

    get_data(config_path = parsed_arg.config)