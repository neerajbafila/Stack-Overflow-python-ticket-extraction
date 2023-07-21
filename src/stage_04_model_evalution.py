from src.Logger.logger import logger
from src.utils import read_config, save_json, create_directories
import os
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import sklearn.metrics as metrics
import math
import argparse

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
            plot_dir = self.config['Plots']['plot_dir']
            create_directories([plot_dir])
            PRC_json_path = self.config["Plots"]["PRC"]
            ROC_json_path = self.config["Plots"]["ROC"]
            ROC_json_full_path = Path(os.path.join(plot_dir, ROC_json_path))
            PRC_json_full_path = Path(os.path.join(plot_dir, PRC_json_path))
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
            self.my_logger.write_log(f"calculating prediction probabilities for auc roc , prediction, recall")
            prediction_prob = loaded_model.predict_proba(X)
            positive_class = prediction_prob[:,1] # [as class 0,1 stored as[0,1]]
            avg_precision = metrics.average_precision_score(label, positive_class)
            roc_auc = metrics.roc_auc_score(label, positive_class)
            score = {
                "avg_precision": avg_precision,
                "roc_auc": roc_auc
            }

            save_json(score_json_path, score, self.my_logger)
            self.my_logger.write_log(f"roc_auc_score has been stored at {metrics_path}")
            # calculate auc_roc_curve
            fpr, tpr, threshold = metrics.roc_curve(label, positive_class)
            roc_points = zip(fpr, tpr, threshold)
            roc_curve_data = {
                "roc_curve_data":[
                    {'fpr': fpr, 'tpr': tpr, 'threshold': threshold} for fpr, tpr, threshold in roc_points 
                ]
            }
            
            save_json(ROC_json_full_path, roc_curve_data, self.my_logger)
            self.my_logger.write_log(f"fpr, tpr, threshold saved at {ROC_json_path}")

            # calculate precision, recall

            precision, recall, threshold_p = metrics.precision_recall_curve(label, positive_class)
            nth_points = math.ceil(len(precision)/1000) # otherwise too many points(arround 6955), so to reduce them taking step size
            prc_points_all =  list(zip(precision, recall, threshold_p))
            # print(len(prc_points_all)) # 6955 points
            prc_points = prc_points_all[::nth_points] # 994 points by taking step size = nth_points=7
            prc_points_data = {
                'prc_points':
                    [{'precision': precision, 'recall': recall, 'threshold': threshold_p} for precision, recall, threshold_p in prc_points]
            }
            
            save_json(PRC_json_full_path, prc_points_data, self.my_logger)



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

        model_eval = Model_evaluation(parsed_args.config, parsed_args.params)
        model_eval.evaluate_model()
        my_logger.write_log(f"{('*')*30}{STAGE} completed{('*')*30}")
    except Exception as e:
        my_logger.write_exception(e)
