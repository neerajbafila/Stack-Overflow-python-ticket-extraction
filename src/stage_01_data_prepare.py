from utils import create_directories, read_config, process_posts
from src.Logger.logger import logger
import os
import argparse
from pathlib import Path


STAGE = "Prepare_data" ## 
class Data_prepare:
    def __init__(self, config_file="config/config.yaml", params="config/params.yaml"):
        self.logger = logger(config_file)

        try:
            self.logger.write_log("Reading config file")
            self.config = read_config(config_file)
            self.params = read_config(params)
        except Exception as e:
            self.logger.write_exception(e)

    def prepare_data(self):
        try:

            source_dir = self.config['Data']['source_dir']
            data_file = self.config['Data']['data_file']
            data_dir_full_path = Path(os.path.join(source_dir, data_file))
            seed = self.params['Prepare']['seed']
            tag = self.params['Prepare']['tag']
            split = self.params['Prepare']['split']
            artifacts_dir = self.config['Artifact']['artifact_dir']
            prepared_data_dir = self.config['Artifact']['prepared_data_dir']
            prepared_data_dir_full_path = Path(os.path.join(artifacts_dir, prepared_data_dir))
            create_directories([prepared_data_dir_full_path])
            train_data_file = self.config['Artifact']['train_data']
            test_data_file = self.config['Artifact']['test_data']
            train_data_file_path = Path(os.path.join(prepared_data_dir_full_path, train_data_file))
            test_data_file_path = Path(os.path.join(prepared_data_dir_full_path, test_data_file))
            encoding = 'utf8'
            column_names = self.params['Prepare']['column_names']
            self.logger.write_log("Writing training and test data")
            with open(data_dir_full_path, 'r', encoding=encoding) as main_data_file:
                with open(train_data_file_path, 'w', encoding=encoding) as train_data_f:
                    with open(test_data_file_path, 'w', encoding=encoding) as test_data_f:
                        try:
                            process_posts(main_data_file, train_data_f, test_data_f, split, column_names, tag, self.logger)
                            self.logger.write_log("Writing training and test data successful")
                        except Exception as e:
                            self.logger.write_exception(e)
            
        except Exception as e:
            self.logger.write_exception(e)
        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--c", default="config/config.yaml")
    parser.add_argument("--params", "--p", default="config/params.yaml")
    parsed_args = parser.parse_args()
    my_logger = logger(parsed_args.config)
    my_logger.write_log(f"{('*')*30}{STAGE} STAGE started{('*')*30}")
    try:

        data_prepare_ob = Data_prepare(parsed_args.config, parsed_args.params)
        data_prepare_ob.prepare_data()
        my_logger.write_log(f"{('*')*30}{STAGE} STAGE completed{('*')*30}")
    except Exception as e:
        my_logger.write_exception(e)
