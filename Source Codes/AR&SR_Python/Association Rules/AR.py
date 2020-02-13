# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import collections as col

class AssosiationRules: 
    '''
    AssosiationRules(pruning=10, session_key='SessionId', item_keys=['ItemId'])
    Parameters
    --------
    pruning : int
        Prune the results per item to a list of the top N co-occurrences. (Default value: 10)
    session_key : string
        The data frame key for the session identifier. (Default value: SessionId)
    item_keys : string
        The data frame list of keys for the item identifier as first item in list 
        and features keys next. (Default value: [ItemId])
    '''
    def __init__( self, pruning=10, session_key='SessionID', item_keys=['ItemID'] ):
        self.pruning = pruning
        self.session_key = session_key
        self.item_keys = item_keys
        self.items_features = {}
        self.predict_for_item_ids = []
        
    def fit( self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. 
            It has one column for session IDs, one for item IDs and many for the
            item features if exist.
            It must have a header. Column names are arbitrary, but must 
            correspond to the ones you set during the initialization of the 
            network (session_key, item_keys).
        '''
        cur_session = -1
        last_items = []
        all_rules = []
        indices_item = []
        for i in self.item_keys:
            all_rules.append(dict())
            indices_item.append( data.columns.get_loc(i) )

        data.sort_values(self.session_key, inplace=True)
        index_session = data.columns.get_loc(self.session_key)
        
        #Create Dictionary of items and their features
        for row in data.itertuples( index=False ):
            item_id = row[indices_item[0]]
            if not item_id in self.items_features.keys() :
                self.items_features[item_id] = []
                for i in indices_item:
                    self.items_features[item_id].append(row[i])
              
        for i in range(len(self.item_keys)):
            rules = all_rules[i]
            index_item = indices_item[i]
            for row in data.itertuples( index=False ):
                session_id, item_id = row[index_session], row[index_item]
                if session_id != cur_session:
                    cur_session = session_id
                    last_items = []
                else: 
                    for item_id2 in last_items:                
                        if not item_id in rules :
                            rules[item_id] = dict()                
                        if not item_id2 in rules :
                            rules[item_id2] = dict()                
                        if not item_id in rules[item_id2]:
                            rules[item_id2][item_id] = 0           
                        if not item_id2 in rules[item_id]:
                            rules[item_id][item_id2] = 0
                        
                        rules[item_id][item_id2] += 1
                        rules[item_id2][item_id] += 1
                        
                last_items.append(item_id)
                
            if self.pruning > 0:
                rules = self.prune(rules) 
                
            all_rules[i] = rules
        self.all_rules = all_rules
        self.predict_for_item_ids = list(self.all_rules[0].keys())
    def predict_next(self, session_items, k = 20):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_items : List
            Items IDs in current session.
        k : Integer
            How many items to recommend
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. 
            Indexed by the item IDs.
        
        '''
        all_len = len(self.predict_for_item_ids)
        input_item_id = session_items[-1]
        preds = np.zeros( all_len ) 
             
        if input_item_id in self.all_rules[0].keys():
            for k_ind in range(all_len):
                key = self.predict_for_item_ids[k_ind]
                if key in session_items:
                    continue
                try:
                    preds[ k_ind ] += self.all_rules[0][input_item_id][key]
                except:
                    pass
                for i in range(1, len(self.all_rules)):
                    input_item_feature = self.items_features[input_item_id][i]
                    key_feature = self.items_features[key][i]
                    try:
                        preds[ k_ind ] += self.all_rules[i][input_item_feature][key_feature]
                    except:
                        pass
        
        series = pd.Series(data=preds, index=self.predict_for_item_ids)
        series = series / series.max()
        
        return series.nlargest(k).index.values
    
    def prune(self, rules): 
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
            --------
            rules : dict of dicts
                The rules mined from the training data
        '''
        for k1 in rules:
            tmp = rules[k1]
            if self.pruning < 1:
                keep = len(tmp) - int( len(tmp) * self.pruning )
            elif self.pruning >= 1:
                keep = self.pruning
            counter = col.Counter( tmp )
            rules[k1] = dict()
            for k2, v in counter.most_common( keep ):
                rules[k1][k2] = v
        return rules