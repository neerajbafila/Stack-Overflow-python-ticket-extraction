from src.Logger.logger import logger
from pathlib import Path
import os
from utils import read_config, create_directories
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import argparse

STAGE = "Model Training"

class ModelTraining:
    def __init__(self, config_file="config/config.yaml", params="config/params.yaml"):
        self.my_logger = logger(config_file)
        try:
           
           self.my_logger.write_log("Reading config file")
           self.config = read_config(config_file)
           self.params = read_config(params)
        except Exception as e:
            self.my_logger.write_exception(e)
    
    def train_model(self):
        try:

            self.my_logger.write_log(f"Reading file path configs")
            artifacts = self.config["Artifact"]['artifact_dir']
            Train_Tfidf_matrix = self.config['Artifact']['Train_Tfidf_matrix']
            featurized_data_dir_path = Path(os.path.join(artifacts, self.config['Artifact']["Featurized_Data_Dir"]))
            Train_Tfidf_matrix_path = Path(os.path.join(featurized_data_dir_path, Train_Tfidf_matrix))
            model_dir = Path(os.path.join(artifacts, self.config['Artifact']['Model_dir']))
            create_directories([model_dir])
            model_name = self.config["Artifact"]['model_name']
            model_full_path = Path(os.path.join(model_dir,model_name))
            seed = self.params['Training']['seed']
            n_est = self.params['Training']['n_est']
            n_jobs = self.params['Training']['n_jobs']
            min_split = self.params['Training']['min_split']
            verbose = self.params['Training']['verbose']
            self.my_logger.write_log(f"Reading configs files done")

        # get training data
            self.my_logger.write_log(f"Getting training data from  {Train_Tfidf_matrix_path}")
            matrix = joblib.load(Train_Tfidf_matrix_path)
            X = matrix[:,2:] # for training we need text
            label = matrix[:,1] # [pid, label, text]
            label = np.squeeze(label.toarray()) # for [0,0,1,0,1] format
            self.my_logger.write_log(f"Training X of shape {X.shape} and label of shape{label.shape} loaded successfully")
            
            model_rf = RandomForestClassifier(random_state=seed, n_estimators=n_est,
                                              n_jobs=n_jobs, min_samples_split=min_split, verbose=verbose)
            self.my_logger.write_log(f"RandomForestClassifier Model Training with below parameters\n\t"
                                     f"seed={seed}, n_est={n_est} min_split={min_split} n_jobs={n_jobs} started")
            model_rf.fit(X, label)
            self.my_logger.write_log(f"Model training completed successfully ")
            joblib.dump(model_rf,model_full_path)
            self.my_logger.write_log(f"Model saved at {model_full_path}")
            
        except Exception as e:
            self.my_logger.write_exception(e)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--c", default="config/config.yaml")
    parser.add_argument("--params", "--p", default="config/params.yaml")
    parsed_args = parser.parse_args()
    my_logger = logger(parsed_args.config)
    my_logger.write_log(f"{('*')*30}{STAGE} started{('*')*30}")
    try:

        model_training_ob = ModelTraining(parsed_args.config, parsed_args.params)
        model_training_ob.train_model()
        my_logger.write_log(f"{('*')*30}{STAGE} completed{('*')*30}")
    except Exception as e:
        my_logger.write_exception(e)
