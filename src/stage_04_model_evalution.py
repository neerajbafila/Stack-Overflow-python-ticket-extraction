from src.Logger.logger import logger
from src.utils import read_config, save_json, create_directories
import os
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import sklearn.metrics as metrics
import json

STAGE = "Model Evaluation"

class Model_evaluation:
    def __init__(self, config_file="config/config.yaml", params="config/params.yaml"):
        self.my_logger = logger(config_file)
        try:
           
           self.my_logger.write_log("Reading config file")
           self.config = read_config(config_file)
           self.params = read_config(params)
        except Exception as e:
            self.my_logger.write_exception(e)
    
    def evaluate_model(self):
        try:
            # get the model
            self.my_logger.write_log(f"Reading file path configs")
            artifacts = self.config["Artifact"]['artifact_dir']
            model_dir = Path(os.path.join(artifacts, self.config['Artifact']['Model_dir']))
            model_name = self.config["Artifact"]['model_name']
            model_full_path = Path(os.path.join(model_dir,model_name))
            Test_Tfidf_matrix = self.config['Artifact']['Test_Tfidf_matrix']
            featurized_data_dir_path = Path(os.path.join(artifacts, self.config['Artifact']["Featurized_Data_Dir"]))
            Test_Tfidf_matrix_path = Path(os.path.join(featurized_data_dir_path, Test_Tfidf_matrix))
            PRC_json_path = self.config["Plots"]["PRC"]
            ROC_json_path = self.config["Plots"]["ROC"]
            score_json = self.config["Metrics"]["scores"]
            metrics_path = self.config["Metrics"]["path"]
            create_directories(metrics_path)
            score_json_path = Path(os.path.join(metrics_path, score_json))
            

            self.my_logger.write_log(f"Reading configs files done")
            self.my_logger.write_log(f"loading model from {model_full_path}")
            loaded_model = joblib.load(model_full_path)
            self.my_logger.write_log(f"model loaded from {model_full_path}")
            # get test data for evaluation
            self.my_logger.write_log(f"getting test data from {Test_Tfidf_matrix_path} for evaluation")
            matrix = joblib.load(Test_Tfidf_matrix_path)
            X = matrix[:,2:] # [id, label, text]
            label = matrix[:,1]
            label = np.squeeze(label.toarray()) # to convert 2d to [1,1,0,1,0] format
            pred = loaded_model.predict(X)
            accuracy = accuracy_score(label, pred)
            confusion_mat = confusion_matrix(label, pred)
            f1 = f1_score(label, pred, pos_label=1)
            self.my_logger.write_log(f"accuracy is {accuracy}")
            self.my_logger.write_log(f"confusion matrix  is {confusion_mat}")

            # calculate prediction probabilities for auc roc

            prediction_prob = loaded_model.predict_proba(X)
            positive_class = prediction_prob[:,1] # [as class 0,1 stored as[0,1]]
            avg_precision = metrics.average_precision_score(label, positive_class)
            roc_auc = metrics.roc_auc_score(label, positive_class)
            score = {
                "avg_precision": avg_precision,
                "roc_auc": roc_auc
            }

            save_json(score_json_path, score, self.my_logger)





        except Exception as e:
            self.my_logger.write_exception(e)

ob = Model_evaluation()
ob.evaluate_model()