import requests
import pandas as pd
from utils import make_data_home, make_file_path, is_valid_path

dataset_directory = 'clustera_datasets'

class Leukemia10_Dataset_Builder:

    def __init__(self):
        self.name = 'leukemia-selected-10.csv'
        self._remote_URL = 'https://raw.githubusercontent.com/kalyaniuniversity/mgx-datasets/master/leukemia/datasets/preprocessed/leukemia-selected-10-snr.csv'
        self.class_labels = ['AML', 'ALL']
        self.class_attribute_name = 'class'
        self.class_label_count = { 'ALL': 47, 'AML': 25 }
        self.attributes_count = 10
        self._file_path = None

    
    def download_prepare(self):
        global dataset_directory
        make_data_home(dataset_directory)

        self._file_path = make_file_path(dataset_directory, self.name)

        if not is_valid_path(self._file_path):

            try:
                data_response = requests.get(self._remote_URL)
                data_response.raise_for_status()

                csvfile = open(self._file_path, 'wb')
                csvfile.write(data_response.content)

            except:
                print(f'\nError occurred!!Status code: {data_response.status_code}')


    def as_dataframe(self):
        global dataset_directory
        return pd.read_csv(self._file_path)


