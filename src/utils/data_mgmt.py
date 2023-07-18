import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import re
import pandas as pd

def process_posts(main_data_file, train_data_file, test_data_file, split, column_names, target_tag, logger):
    column_names = column_names
    train_data_file.write(column_names)
    test_data_file.write(column_names)
    line_no = 1
    
    for line in tqdm(main_data_file):
        try: 
            writer = train_data_file if random.random() > split else test_data_file
            attr = ET.fromstring(line).attrib
            pid = attr.get("Id")
            label = 1 if target_tag in attr.get("Tags", "") else 0
            title = re.sub(r"s\+"," ",attr.get("Title", "")).strip()
            body = re.sub(r"\s+", " ", attr.get("Body", "")).strip()
            text = f"{title} {body}"
            writer.write(f"{pid}\t{label}\t{text}\n")
            line_no +=1
            
        except Exception as e:
            print(f"Skipping the broken line {line_no}: {e}\n")
            logger.write_exception(f"Skipping the broken line {line_no}: {e}\n ")
            
def get_data_as_df(logger, source_data_path:str, encoding='utf8', sep='\t') -> pd.DataFrame:
    try:
        df = pd.read_csv(source_data_path, encoding=encoding, delimiter=sep)
        logger.write_log(f"Data has been read from {source_data_path} as pandas dataFrame")
        return df
    except Exception as e:
        logger.write_exception(e)





