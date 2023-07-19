import argparse
import os
import numpy as np
from tqdm import tqdm
import logging
from src.utils import read_config, create_directories, get_data_as_df, save_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from src.Logger.logger import logger
from pathlib import Path

STAGE = "Featurization" ## <<< change stage name 

class Featurization:
    def __init__(self, config_file="config/config.yaml", params="config/params.yaml"):
        self.my_logger = logger(config_file)
        try:
           
           self.my_logger.write_log("Reading config file")
           self.config = read_config(config_file)
           self.params = read_config(params)
        except Exception as e:
            self.my_logger.write_exception(e)


    def featurization(self):
        try:

            ## read config files
            self.my_logger.write_log(f"Reading file path configs")
            artifacts = self.config["Artifact"]['artifact_dir']
            prepare_data_dir = self.config["Artifact"]['prepared_data_dir']
            prepare_data_dir_path = Path(os.path.join(artifacts, prepare_data_dir))
            

            train_data_path = os.path.join(prepare_data_dir_path,self.config['Artifact']["train_data"])
            test_data_path = os.path.join(prepare_data_dir_path,self.config['Artifact']["test_data"])

            featurized_data_dir_path = Path(os.path.join(artifacts, self.config['Artifact']["Featurized_Data_Dir"]))
            create_directories([featurized_data_dir_path])

            featurized_train_data_path = Path(os.path.join(featurized_data_dir_path, self.config['Artifact']["Featurized_Train_Data"]))
            featurized_test_data_path = Path(os.path.join(featurized_data_dir_path, self.config['Artifact']["Featurized_Test_Data"]))
            Train_Tfidf_matrix = self.config['Artifact']['Train_Tfidf_matrix']
            Train_Tfidf_matrix_path = Path(os.path.join(featurized_data_dir_path, Train_Tfidf_matrix))
            Test_Tfidf_matrix = self.config['Artifact']['Test_Tfidf_matrix']
            Test_Tfidf_matrix_path = Path(os.path.join(featurized_data_dir_path, Test_Tfidf_matrix))

            max_features = self.params["featurization"]["max_features"]
            n_grams = self.params["featurization"]["ngram_range"]
            self.my_logger.write_log(f"Reading config files successfully done")
            
            # train

            self.my_logger.write_log(f"getting pandas data frame")
            df_train = get_data_as_df(self.my_logger, train_data_path)
            self.my_logger.write_log(f"getting pandas data frame successfully done")
            df_train.to_csv(featurized_train_data_path, sep="\t")
            self.my_logger.write_log(f"training pandas data frame is saved at {featurized_train_data_path}")
            self.my_logger.write_log(f"converting data to text format")
            train_words = np.array(df_train.text.str.lower().values.astype("U"))

            self.my_logger.write_log(f"creating bag of words using CountVectorizer with max_features={max_features} and ngrams={n_grams}")
            bag_of_words = CountVectorizer(
                stop_words="english",
                max_features=max_features,
                ngram_range=(1, n_grams)
            )

            bag_of_words.fit(train_words)
            train_words_binary_matrix = bag_of_words.transform(train_words)
            self.my_logger.write_log(f"bag of words successfully created")

            self.my_logger.write_log(f"creating TFIDF using TfidfTransformer")
            tfidf = TfidfTransformer(smooth_idf=False)
            tfidf.fit(train_words_binary_matrix)
            train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)
            self.my_logger.write_log(f"creating TFIDF successfully done")

            # call a function to save this matrix
            save_matrix(dataframe=df_train, matrix=train_words_tfidf_matrix, output_path=Train_Tfidf_matrix_path, logger=self.my_logger)

            # for test data
            self.my_logger.write_log(f"***************Doing for test data***************")
            self.my_logger.write_log(f"getting pandas data frame for test data")
            df_test = get_data_as_df(self.my_logger, test_data_path)
            df_test.to_csv(featurized_test_data_path)
            self.my_logger.write_log(f"testing pandas data frame is saved at {featurized_train_data_path}")
            self.my_logger.write_log(f"converting data into text format for Test data")
            test_words = np.array(df_test.text.str.lower().values.astype("U"))
            self.my_logger.write_log(f"creating Bag of words for Test Data")
            test_words_binary_matrix = bag_of_words.transform(test_words)
            self.my_logger.write_log(f"creating TFIDF for Test Data")
            test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
            # call a function to save this matrix
            save_matrix(dataframe=df_test, matrix=test_words_tfidf_matrix, output_path=Test_Tfidf_matrix_path, logger=self.my_logger)
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

        Featurization_ob = Featurization(parsed_args.config, parsed_args.params)
        Featurization_ob.featurization()
        my_logger.write_log(f"{('*')*30}{STAGE} completed{('*')*30}")
    except Exception as e:
        my_logger.write_exception(e)
