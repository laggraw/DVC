from pandas.core.indexes.base import Index
from src.utils.all_utils import read_yaml, create_directory,save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_save(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # save the dataset in local directory
    # create directory path : artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"] 
    raw_local_dir = config["artifacts"]["raw_local_dir"] 
    raw_local_file = config["artifacts"]["raw_local_file"]

    raw_local_file_path = os.path.join(artifacts_dir, raw_local_dir,raw_local_file)
    # print(raw_local_file_path)
    df = pd.read_csv(raw_local_file_path)
    # print(df.head(10))

    random_state = params['base']['random_state']
    test_size = params['base']['random_state']
    # shuffle = params['base']['shuffle']

    train, test = train_test_split(df, test_size = test_size, random_state = random_state)

    split_data_dir = config["artifacts"]["split_data_dir"] 
    train_path_filename = config["artifacts"]["train"] 
    test_path_filename = config["artifacts"]["test"] 

    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    train_data_path = os.path.join(artifacts_dir , split_data_dir, train_path_filename)
    test_data_path = os.path.join(artifacts_dir , split_data_dir, test_path_filename)

    
    create_directory(dirs = [split_data_dir_path ])

    save_local_df(train,train_data_path)
    save_local_df(test,test_data_path)

if __name__ =='__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_arg = args.parse_args()

    print(parsed_arg.config)

    split_save(config_path = parsed_arg.config, params_path = parsed_arg.params)