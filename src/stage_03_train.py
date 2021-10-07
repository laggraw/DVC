from pandas.core.indexes.base import Index
from src.utils.all_utils import read_yaml, create_directory,save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet,LinearRegression
import joblib


def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
  
    # create directory path : artifacts/split_data_dir/train.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"] 
    split_data_dir = config["artifacts"]["split_data_dir"] 
    train_path_filename = config["artifacts"]["train"] 
    # test_path_filename = config["artifacts"]["test"] 

    random_state = params['base']['random_state']
    alpha_value = params['model_param']['ElasticNet']['alpha']
    l1_ratio_value = params['model_param']['ElasticNet']['l1_ratio']

    train_data_path = os.path.join(artifacts_dir , split_data_dir, train_path_filename)
    # test_data_path = os.path.join(artifacts_dir , split_data_dir, test_path_filename)

    train_data = pd.read_csv(train_data_path)
    train_y = train_data['quality']
    train_x = train_data.drop('quality', axis=1)

    model_dir = config["artifacts"]["model_dir"]
    En_model_filename = config["artifacts"]["En_model_file"]
    Lr_model_filename = config["artifacts"]["Lr_model_file"]


    model_dir_path = os.path.join(artifacts_dir, model_dir)
    En_model_file_path = os.path.join(artifacts_dir, model_dir, En_model_filename)
    Lr_model_file_path = os.path.join(artifacts_dir, model_dir, Lr_model_filename)
    create_directory(dirs =[model_dir_path])
 

    en = ElasticNet(alpha=alpha_value, l1_ratio= l1_ratio_value, random_state=random_state)
    en.fit(train_x, train_y)
    joblib.dump(en,En_model_file_path)

    lr = LinearRegression()
    lr.fit(train_x, train_y)

    joblib.dump(lr,Lr_model_file_path)

    print("training done")

if __name__ =='__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_arg = args.parse_args()

    print(parsed_arg.config)

    train_model(config_path = parsed_arg.config, params_path = parsed_arg.params)