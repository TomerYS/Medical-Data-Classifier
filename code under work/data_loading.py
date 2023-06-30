import pandas as pd
import numpy as np

def load_datasets():
    personal_info_train = pd.read_csv('personal_info_train.csv')
    measurements_train = pd.read_csv('measurements_results_train.csv')
    personal_info_test = pd.read_csv('personal_info_test.csv')
    measurements_test = pd.read_csv('measurements_results_test.csv')
    print('Datasets loaded successfully')

    return personal_info_train, measurements_train, personal_info_test, measurements_test
