# coding:utf-8
from __future__ import absolute_import
import tensorflow as tf
import gc
import os
from csrm import CSRM
import argparse
#import data_process
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run CSRM.")
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--n_items', type=int, default=39164, help='Item size')
    parser.add_argument('--dim_proj', type=int, default=100, help='Item embedding dimension. initial:50')
    parser.add_argument('--hidden_units', type=int, default=100, help='Number of GRU hidden units. initial:100')
    parser.add_argument('--display_frequency', type=int, default=5, help='Display to stdout the training progress every N updates.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--keep_probability', nargs='?', default='[0.75,0.5]', help='Keep probability (i.e., 1-dropout_ratio). 1: no dropout.')
    parser.add_argument('--no_dropout', nargs='?', default='[1.0,1.0]', help='Keep probability (i.e., 1-dropout_ratio). 1: no dropout.')
    parser.add_argument('--memory_size', type=int, default=512, help='.')
    parser.add_argument('--memory_dim', type=int, default=100, help='.')
    parser.add_argument('--shift_range', type=int, default=1, help='.')
    parser.add_argument('--controller_layer_numbers', type=int, default=0, help='.')
    parser.add_argument('--K', type=int, default=20, help='evaluation K recall@K and MRR@K')
    parser.add_argument('--itemid', default='itemId', type=str)
    parser.add_argument('--sessionid', default='sessionId', type=str)
    parser.add_argument('--valid_data', default='Digintica_test.csv', type=str)
    parser.add_argument('--train_data', default='2-a-2(intermediate).csv', type=str)
	parser.add_argument('--expname', default='experimentName', type=str)
    parser.add_argument('--data_folder', default='C://Users//s-moh//0-Labwork//Rakuten Project//Dataset//Benchmarking//Digintica', type=str)
    return parser.parse_args()

def load_sequence(from_path, itemid, sessionid, Train = True, itemsIDs = [], old_new = {}):
    freqs = {}
    data = pd.read_csv(from_path)
    if Train == True:
        itemsIDs = list(data[itemid].unique())
        data[itemid] = data[itemid].astype('category')
        new_old = dict(enumerate(data[itemid].cat.categories))
        old_new = {y:x for x,y in new_old.items()}
        data[['tmp']] = data[[itemid]].apply(lambda x: x.cat.codes+1)
        freqs = dict(data['tmp'].value_counts())
        
    patterns = []
    labels = []
    cnt_session = -1
    cnt_pattern = []
    for i in range(len(data)):
        if i % 100000 == 0:
            print('Finished Till Now: ', i)
        sid = data.loc[i, [sessionid]][0]
        iid = data.loc[i, [itemid]][0]
        if sid != cnt_session:
            cnt_session = sid
            cnt_pattern = []
        if Train == False and iid not in itemsIDs:
            continue
        cnt_pattern.append(old_new[iid]+1 )        
        if len(cnt_pattern) > 1:
            lst_pattern = []
            if len(patterns) > 0:
                lst_pattern = patterns[-1]
            if cnt_pattern != lst_pattern:
                patterns.append(cnt_pattern[:-1])
                labels.append(cnt_pattern[-1])
    
    return (patterns, labels), itemsIDs, freqs, old_new

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    args = parse_args()
	gc.collect()
	
	print('Train:', args.train_data, ' -- Test:', args.valid_data)
	with open("LOGGER_"+ args.expname + ".txt", "a") as myfile:
		myfile.write(row['train'] + ", " + row['test'] +"\n")
	
	# split patterns to train_patterns and test_patterns
	print('Start Data Preprocessing: Training Set')
	train, itemsIDs, freqs, old_new = load_sequence(args.data_folder + '/' + args.train_data, 
													args.itemid, args.sessionid, 
													itemsIDs = [])
	args.n_items = len(itemsIDs) + 1
	print('Start Data Preprocessing: Testing Set')
	valid, _, _, _ = load_sequence(args.data_folder + '/' + args.valid_data, 
								   args.itemid, args.sessionid, Train = False, 
								   itemsIDs = itemsIDs, 
								   old_new = old_new)

	#train, valid, test = data_process.load_data()
	print("%d train examples." % len(train[0]))
	print("%d valid examples." % len(valid[0]))
	keep_probability = np.array(args.keep_probability)
	no_dropout = np.array(args.no_dropout)
	result_path = "./save/" + args.dataset
	# Build model
	tf.reset_default_graph()
	with tf.Session(config=config) as sess:
		model = CSRM(sess=sess, n_items=args.n_items, dim_proj=args.dim_proj,
			hidden_units=args.hidden_units, memory_size=args.memory_size,
			memory_dim=args.memory_dim, shift_range=args.shift_range, 
			lr=args.lr, controller_layer_numbers=args.controller_layer_numbers, 
			batch_size=args.batch_size, keval = args.K, epoch=args.epoch, 
			keep_probability=keep_probability, no_dropout=no_dropout,
			display_frequency=args.display_frequency, item_freqs = freqs, 
			expname = args.expname)
		hit, MRR, cov, pop, train_time, test_time = model.train(train, valid, valid, result_path)

	print('Recall:', hit)
	print ('\nMRR:', MRR)
	print ('\nCoverage:', cov)
	print ('\nPopularity:', pop)
	print ('\ntrain_time:', train_time)
	print ('\ntest_time:', test_time)
	print('End Model Predictions')

if __name__ == '__main__':
    main()