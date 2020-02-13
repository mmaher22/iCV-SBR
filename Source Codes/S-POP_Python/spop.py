# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
  
class SessionPop:
    '''
    SessionPop(top_n=100, item_key='ItemId', support_by_key=None)
    Session popularity predictor that gives higher scores to items with higher number of occurrences in the session. 
    Ties are broken up by adding the popularity score of the item.
    The score is given by:
    .. math::
        r_{s,i} = supp_{s,i} + \\frac{supp_i}{(1+supp_i)}
    Parameters
    --------
    top_n : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    '''    
    def __init__(self, top_n = 1000, session_key = 'SessionId', item_key = 'ItemId'):
        self.top_n = top_n
        self.item_key = item_key
        self.session_id = session_key
        
    def fit(self, data):
        '''
        Trains the predictor.
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. 
            It has one column for session IDs, one for item IDs.
        '''
        self.items = data[self.item_key].unique()
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)
        self.prev_session_id = -1
         
    def predict_next(self, last_items, k):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
        --------
        last_items : list of items clicked in current session
        k : number of items to recommend and evaluate based on it
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        '''
        pers = {}
        for i in last_items:
            pers[i] = pers[i] + 1 if i in pers.keys() else  1
        
        preds = np.zeros(len(self.items))
        mask = np.in1d(self.items, self.pop_list.index)
        ser = pd.Series(pers)
        preds[mask] = self.pop_list[self.items[mask]]
        
        mask = np.in1d(self.items, ser.index)
        preds[mask] += ser[self.items[mask]]
        
        series = pd.Series(data=preds, index=self.items)
        series = series / series.max()    
        return series.nlargest(k).index.values