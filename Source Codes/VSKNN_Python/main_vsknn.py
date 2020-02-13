# -*- coding: utf-8 -*-
import gc
import os
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from vsknn import VMContextKNN

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--neighbors', type=int, default=200, help="K neighbors to be used in KNN")
parser.add_argument('--sample', type=int, default=0, help="Max Number of steps to walk back from the currently viewed item")
parser.add_argument('--weight_score', type=str, default='div_score', help="Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic)_score")
parser.add_argument('--weighting', type=str, default='div', help="Decay function to determine the importance/weight of individual actions in the current session(linear, same, div, log, qudratic)")
parser.add_argument('--similarity', type=str, default='cosine', help="String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: cosine)")
parser.add_argument('--itemid', default='ItemID', type=str)
parser.add_argument('--sessionid', default='SessionID', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--data_folder', default='C://Users//s-moh//0-Labwork//Rakuten Project//Dataset//RecSys_Dataset_After//', type=str)
args = parser.parse_args()

args.expname = row['name']
args.sessionid = row['sessionid']
args.itemid = row['itemid']
args.data_folder = row['path']
args.valid_data = row['test']
args.train_data = row['train']
args.freq = row['freq']

print('Train:', args.train_data, ' -- Test:', args.valid_data, ' -- Freq:', args.freq)
#with open("LOGGER_"+ args.expname + ".txt", "a") as myfile:
#	myfile.write(row['train'] + ", " + row['test'] +"\n")

# Get the arguments
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
x_train.sort_values(args.sessionid, inplace=True)
distinct_train = x_train[args.itemid].nunique()

valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

print('Finished Reading Data \nStart Model Fitting...')
# Fitting Model
t1 = time.time()
#command_memory ="python memoryLogger.py "+ str(cPid) + " " + args.expname+'train'
#memory_task = subprocess.Popen(command_memory, stdout=subprocess.PIPE, shell=True)
model = VMContextKNN(k = args.neighbors, sample_size = args.sample, similarity = args.similarity, 
					 weighting = args.weighting, weighting_score = args.weight_score,
					 session_key = args.sessionid, item_key = args.itemid)
model.fit(x_train)
#memory_task.kill()
train_time = time.time() - t1
print('End Model Fitting\n Start Predictions...')

# Test Set Evaluation
test_size = 0.0
hit = [0.0]
MRR = [0.0]
cov = [[]]
pop = [[]]
Ks = [args.K]
cur_length = 0
cur_session = -1
last_items = []
t1 = time.time()
#command_memory ="python memoryLogger.py "+ str(cPid) + " " + args.expname + "test"
#memory_task = subprocess.Popen(command_memory, stdout=subprocess.PIPE, shell=True)
index_item = x_valid.columns.get_loc(args.itemid)
index_session = x_valid.columns.get_loc(args.sessionid)
train_items = model.items_ids
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
		#print(item_id, item_id in train_items)
		item_id = model.old_new[item_id]
		if len(last_items) > cur_length: #make prediction
			cur_length += 1
			test_size += 1
			# Predict the most similar items to items
			for k in range(len(Ks)):
				predictions = model.predict_next(last_items, k = Ks[k])
				# Evaluation
				rank = 0
				for predicted_item in predictions:
					if predicted_item not in cov[k]:
						cov[k].append(predicted_item)
					pop[k].append(model.freqs[predicted_item])
					rank += 1
					if predicted_item == item_id:
						hit[k] += 1.0
						MRR[k] += 1/rank
						break
		
		last_items.append(item_id)
#memory_task.kill()
hit[:] = [x / test_size for x in hit]
MRR[:] = [x / test_size for x in MRR]
cov[:] = [len(x) / distinct_train for x in cov]
maxi = max(model.freqs.values())
pop[:] = [np.mean(x) / maxi for x in pop]
test_time = (time.time() - t1)
print('Recall:', hit)
print ('\nMRR:', MRR)
print ('\nCoverage:', cov)
print ('\nPopularity:', pop)
print ('\ntrain_time:', train_time)
print ('\ntest_time:', test_time)
print('End Model Predictions')