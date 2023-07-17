import os
from pathlib import Path 
import yaml
from datetime import datetime
import sys

def create_directories(dirs: list):
    if type(dirs) != list:
        dirs = [dirs]
    try:
        full_dir_path = ""
        for dir in dirs:
            full_dir_path = Path(os.path.join(full_dir_path, dir))
        os.makedirs(full_dir_path, exist_ok=True)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_no = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"Exception occurred \nexc_type {exc_type}, exc_obj {exc_obj}, line_no {line_no}, file_name {file_name}")


def read_config(config_file="config/config.yaml"):
    with open(config_file, 'r') as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def get_unique_name():
    now = datetime.now()
    name = now.strftime("%y-%m-%d")
    return name
