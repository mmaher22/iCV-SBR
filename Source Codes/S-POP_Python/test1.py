import os
import time
import argparse
import pandas as pd
from spop import SessionPop

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--topn', type=int, default=100, help="Number of top items to return non zero scores for them (most popular)")
parser.add_argument('--itemid', default='ItemID', type=str)
parser.add_argument('--sessionid', default='SessionID', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--data_folder', 
                    default='/home/icvuser/Desktop/Recsys cleaned data/RecSys15 Dataset Splits', type=str)

# Get the arguments
args = parser.parse_args()
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

print(x_train)