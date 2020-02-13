import os
import time
import argparse
import pandas as pd
from smf import SessionMF

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--factors', type=int, default=100, help="Number of latent factors.")
parser.add_argument('--batch', type=int, default=32, help="Batch size for the training process")
parser.add_argument('--momentum', type=float, default=0.0, help="Momentum of the optimizer adagrad_sub")
parser.add_argument('--regularization', type=float, default=0.0001, help="Regularization Amount of the objective function")
parser.add_argument('--dropout', type=float, default=0.0, help="Share of items that are randomly discarded from the current session while training")
parser.add_argument('--skip', type=float, default=0.0, help="Probability that an item is skiped and the next one is used as the positive example")
parser.add_argument('--neg_samples', type=int, default=2048, help="Number of items that are sampled as negative examples")
parser.add_argument('--activation', type=str, default='linear', help="Final activation function (linear, sigmoid, uf_sigmoid, hard_sigmoid, relu, softmax, softsign, softplus, tanh)")
parser.add_argument('--objective', type=str, default='bpr_max', help="Loss Function (bpr_max, top1_max, bpr, top1)")
parser.add_argument('--epochs', type=int, default=10, help="Number of Epochs")
parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")

parser.add_argument('--itemid', default='ItemID', type=str)
parser.add_argument('--sessionid', default='SessionID', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--data_folder', default='/home/icvuser/Desktop/Recsys cleaned data/RecSys15 Dataset Splits', type=str)

# Get the arguments
args = parser.parse_args()
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
x_train.sort_values(args.sessionid, inplace=True)
x_train = x_train.iloc[-int(len(x_train) / 64) :]   #just take 1/64 last instances

valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

print('Finished Reading Data \nStart Model Fitting...')
# Fitting  Model
t1 = time.time()
model = SessionMF(factors = args.factors, session_key = args.sessionid, item_key = args.itemid, 
                  batch = args.batch, momentum = args.momentum, regularization = args.regularization, 
                  dropout = args.dropout, skip = args.skip, samples = args.neg_samples, 
                  activation = args.activation, objective = args.objective, epochs = args.epochs, learning_rate = args.lr)
model.fit(x_train)
t2 = time.time()
print('End Model Fitting with total time =', t2 - t1, '\n Start Predictions...')

# Test Set Evaluation
test_size = 0.0
hit = 0.0
MRR = 0.0
cur_length = 0
cur_session = -1
last_items = []
t1 = time.time()
index_item = x_valid.columns.get_loc(args.itemid)
index_session = x_valid.columns.get_loc(args.sessionid)
train_items = model.unique_items
counter = 0
for row in x_valid.itertuples( index=False ):
    counter += 1
    if counter % 10000 == 0:
        print('Finished Prediction for ', counter, 'items.')
    session_id, item_id = row[index_session], row[index_item]
    if session_id != cur_session:
        cur_session = session_id
        last_items = []
        cur_length = 0
    
    if item_id in model.item_map.keys():
        if len(last_items) > cur_length: #make prediction
            cur_length += 1
            test_size += 1
            # Predict the most similar items to items
            predictions = model.predict_next(last_items, K = args.K)
            # Evaluation
            rank = 0
            for predicted_item in predictions:
                #print(predicted_item, item_id, '###')
                rank += 1
                if int(predicted_item) == item_id:
                    hit += 1.0
                    MRR += 1/rank
                    break
        
        last_items.append(item_id)
t2 = time.time()
print('Recall: {}'.format(hit / test_size))
print ('\nMRR: {}'.format(MRR / test_size))
print('End Model Predictions with total time =', t2 - t1)