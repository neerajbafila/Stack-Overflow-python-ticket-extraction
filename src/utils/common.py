import os
from pathlib import Path 
import yaml
from datetime import datetime

def create_directories(dirs: list):
    full_dir_path = ""
    for dir in dirs:
        full_dir_path = Path(os.path.join(full_dir_path, dir))
    os.makedirs(full_dir_path, exist_ok=True)


def read_config(config_file="config/config.yaml"):
    with open(config_file, 'r') as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def get_unique_name():
    now = datetime.now()
    name = now.strftime("%y-%m-%d")
    return name
