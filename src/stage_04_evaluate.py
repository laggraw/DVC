from pandas.core.indexes.base import Index
from src.utils.all_utils import read_yaml, create_directory,save_reports
import argparse
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def evaluate_metrics(actual_value, predicted_value):
    rmse = np.sqrt(mean_squared_error(actual_value, predicted_value))
    mae = mean_absolute_error(actual_value, predicted_value)
    r2 = r2_score(actual_value, predicted_value)

    return rmse,mae,r2

def evaluate_model(config_path):
    config = read_yaml(config_path)
  
    # create directory path : 
    artifacts_dir = config["artifacts"]["artifacts_dir"] 
    split_data_dir = config["artifacts"]["split_data_dir"] 
    test_path_filename = config["artifacts"]["test"] 
    # artifacts/split_data_dir/test.csv
    test_data_path = os.path.join(artifacts_dir , split_data_dir, test_path_filename)

    #sperating target and features
    test_data = pd.read_csv(test_data_path)
    test_y = test_data['quality']
    test_x = test_data.drop('quality', axis=1)

    #creating model directory for both model ElasticNet and LinerRegression
    model_dir = config["artifacts"]["model_dir"]
    En_model_filename = config["artifacts"]["En_model_file"]
    Lr_model_filename = config["artifacts"]["Lr_model_file"]

    #creating Directory for reporting
    scores_path_dir = config["artifacts"]["reports_dir"]
    en_scores_filename = config["artifacts"]["en_scores"]
    lr_scores_filename = config["artifacts"]["lr_scores"]

    scores_dir_path = os.path.join(artifacts_dir, scores_path_dir)
    #Report file path for ElasticNet and LinearRegression
    en_scores_file_path = os.path.join(artifacts_dir, scores_path_dir,en_scores_filename)
    lr_scores_file_path = os.path.join(artifacts_dir, scores_path_dir,lr_scores_filename)
    create_directory(dirs =[scores_dir_path ])
    
    # Model file path
    En_model_file_path = os.path.join(artifacts_dir, model_dir, En_model_filename)
    Lr_model_file_path = os.path.join(artifacts_dir, model_dir, Lr_model_filename)
    
    #Loading the models
    en = joblib.load(En_model_file_path)
    lr = joblib.load(Lr_model_file_path)

    #Evaluating Models performance
    y_predict = en.predict(test_x)
    en_rmse,en_mae, en_r2 = evaluate_metrics(test_y,y_predict )
    print(en_rmse,en_mae, en_r2)

    en_scores = {
        "rmse" : en_rmse,
        "mae" : en_mae,
        "r2" : en_r2
    }

    #Saving performance report for ElasticNet Model
    save_reports(en_scores, en_scores_file_path)

    y_predict = lr.predict(test_x)
    lr_rmse,lr_mae, lr_r2 = evaluate_metrics(test_y,y_predict )
    print( lr_rmse,lr_mae, lr_r2)
    lr_scores = {
        "rmse" : lr_rmse,
        "mae" : lr_mae,
        "r2" : lr_r2
    }
    #Saving performance report for LinerRegression Model
    save_reports(lr_scores, lr_scores_file_path)
    print("training done")

if __name__ =='__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    
    parsed_arg = args.parse_args()

    print(parsed_arg.config)

    evaluate_model(config_path = parsed_arg.config)