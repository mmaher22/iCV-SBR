import os
import time
import argparse
import pandas as pd
from AR import AssosiationRules

parser = argparse.ArgumentParser()
parser.add_argument('--prune', type=int, default=10, help="Association Rules Pruning Parameter")
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--itemid', default='ItemID', type=str)
parser.add_argument('--sessionid', default='SessionID', type=str)
parser.add_argument('--item_feats', default='', type=str, 
                    help="Names of Columns containing items features separated by #")
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--data_folder', default='/Dataset/RecSys_Dataset_After/', type=str)

# Get the arguments
args = parser.parse_args()
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

items_feats = [args.itemid]
ffeats = args.item_feats.strip().split("#")
if ffeats[0] != '':
    items_feats.extend(ffeats)

print('Finished Reading Data \nStart Model Fitting...')
# Fitting AR Model
t1 = time.time()
model = AssosiationRules(session_key = args.sessionid, item_keys = items_feats, pruning=args.prune)
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
train_items = model.items_features.keys()
counter = 0
for row in x_valid.itertuples( index=False ):
    counter += 1
    if counter % 5000 == 0:
        print('Finished Prediction for ', counter, 'items.')
    session_id, item_id = row[index_session], row[index_item]
    if session_id != cur_session:
        cur_session = session_id
        last_items = []
        cur_length = 0
    
    if not item_id in last_items and item_id in train_items:
        if len(last_items) > cur_length: #make prediction
            cur_length += 1
            test_size += 1
            # Predict the most similar items to items
            predictions = model.predict_next(last_items, k = args.K)
            #print('preds:', predictions)
            # Evaluation
            rank = 0
            for predicted_item in predictions:
                rank += 1
                if predicted_item == item_id:
                    hit += 1.0
                    MRR += 1/rank
                    break
        
        last_items.append(item_id)
t2 = time.time()
print('Recall: {}'.format(hit / test_size))
print ('\nMRR: {}'.format(MRR / test_size))
print('End Model Predictions with total time =', t2 - t1)