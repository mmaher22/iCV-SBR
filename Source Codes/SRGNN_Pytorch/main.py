# -*- coding: utf-8 -*-
import os
import argparse
import time
import pandas as pd
from utils import Data
from model import SessionGraph, trans_to_cuda, train_test

parser = argparse.ArgumentParser()
## Input Parameters
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--itemid', default='ItemID', type=str)
parser.add_argument('--sessionid', default='SessionID', type=str)
parser.add_argument('--item_feats', default='', type=str, help="Names of Columns containing items features separated by #")
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument('--train_data', default='recSys15Valid.txt', type=str)
#parser.add_argument('--train_data', default='recSysValid_Feature.csv', type=str)
parser.add_argument('--data_folder', default='../Dataset/RecSys_Dataset_After/', type=str)
## Architecture Parameters
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
opt = parser.parse_args()

def preprocess(x_train, x_test, feats_columns, opt):
    unique_items = x_train[opt.itemid].unique()
    item_mapping = {}
    item_features = {0:[0] * len(feats_columns)}
    #Features normalization (Min-Max Scaling)
    for feature in feats_columns:
        x, y = x_train[feature].mean(), x_train[feature].std()
        x_train[feature] = (x_train[feature] - x) / y #standardize features
        x_test[feature] = (x_test[feature] - x) / y #standardize features
        
    train_data = []
    train_data_target = []
    test_data = []
    test_data_target = []
    #map each item ID to a new ID starting from 1
    for item_key, counter in zip(unique_items, range(1, len(unique_items)+1 )):
        item_mapping[item_key] = counter
    #add sesssions data to train_data and test_data
    #TRAIN
    cnt_session_items = []
    cnt_session_id = -1
    for i, row in x_train.iterrows():
        if row[opt.sessionid] != cnt_session_id:
            cnt_session_id = row[opt.sessionid]
            cnt_session_items = []
        if len(cnt_session_items) > 0:
            train_data.append(cnt_session_items)
            train_data_target.append(item_mapping[row[opt.itemid]])
        cnt_session_items.append(item_mapping[row[opt.itemid]])
        
        if not item_mapping[row[opt.itemid]] in item_features.keys():
            feats = []
            for feat in feats_columns:
                feats.append(row[feat])
            item_features[item_mapping[row[opt.itemid]]] = feats
    del x_train
    #TEST
    cnt_session_items = []
    cnt_session_id = -1
    for i, row in x_test.iterrows():
        if row[opt.sessionid] != cnt_session_id:
            cnt_session_id = row[opt.sessionid]
            cnt_session_items = []
        if row[opt.itemid] in unique_items:
            itemid = item_mapping[row[opt.itemid]]
        else:
            itemid = 0
        if len(cnt_session_items) > 0:
            test_data.append(cnt_session_items)
            test_data_target.append(itemid)
        cnt_session_items.append(itemid)
    del x_test
    #print(item_features[0], feats_columns, '----')
    return [train_data, train_data_target], [test_data, test_data_target], len(unique_items) + 1, item_features

def main():
    #Read and sort the data
    train_data = os.path.join(opt.data_folder, opt.train_data)
    x_train = pd.read_csv(train_data)
    x_train.sort_values(opt.sessionid, inplace=True)
    x_train = x_train.iloc[-int(len(x_train) / 4):]
    valid_data = os.path.join(opt.data_folder, opt.valid_data)
    x_valid = pd.read_csv(valid_data)
    x_valid.sort_values(opt.sessionid, inplace=True)
    #Extract Item Features column names (if exists)
    feats_columns = []
    ffeats = opt.item_feats.strip().split("#")
    if ffeats[0] != '':
        feats_columns.extend(ffeats)
        
    train_data, test_data, n_node, item_features = preprocess(x_train, x_valid, feats_columns, opt)
    del x_train
    del x_valid
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    print('Number of Nodes:', n_node)
    model = trans_to_cuda(SessionGraph(opt, n_node, item_features))
    start = time.time()
    best_result = [0, 0]
    #bad_counter = 0
    f=open("outputlogs.txt","a+")
    print('START TRAINING........')
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('Epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data, opt.K)
        #flag = 0
        if hit >= best_result[0] or mrr >= best_result[1]:
            #flag = 1
            best_result[0] = hit
            best_result[1] = mrr
            
        print('Epoch%d: \tRecall@%d:\t%.4f\tMMR@%d:\t%.4f'% (epoch, opt.K, hit, opt.K, mrr))
        f.write('Epoch%d: \tRecall@%d:\t%.4f\tMMR@%d:\t%.4f'% (epoch, opt.K, hit, opt.K, mrr))
        
        #bad_counter += 1 - flag
        #if bad_counter >= opt.patience:
        #    break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    f.write("\nRun time: %f s" % (end - start))
    f.write('\n-----------------------------------------------------------')
    f.close()

if __name__ == '__main__':
    main()
