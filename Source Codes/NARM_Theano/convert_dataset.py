import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

#data config (all methods)
DATA_PATH = 'D:/Projects/Others/sessionRec_NARM/Datasets/rsc15-raw/'
DATA_PATH_PROCESSED = 'D:/Projects/Others/sessionRec_NARM/Datasets/rsc15-raw/processed'
DATA_FILE = '/rsc15-clicks'

def preprocess_org( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED):
    '''
        Preprocessing from original gru4rec project
    '''
    input_file = path + file
    output_file = path_proc + file

    #load csv
    data = pd.read_csv( input_file + '.dat' , sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
    data.columns = ['SessionId', 'Time', 'ItemId']

    data.head(10000).to_csv( output_file + '_first_1000.txt', sep=',', index=False)

if __name__ == '__main__':
    preprocess_org()