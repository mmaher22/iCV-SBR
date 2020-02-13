from _operator import itemgetter
from math import sqrt
import time
import numpy as np
import pandas as pd
from math import log10

class VMContextKNN:
    '''
    VMContextKNN( k, sample_size=1000, similarity='cosine', weighting='div', weighting_score='div_score', session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 200)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 2000)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: cosine)
    weighting : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    weighting_score : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    '''
    def __init__( self, k=200, sample_size=0, similarity='cosine', weighting='div', weighting_score='div_score', session_key = 'SessionId', item_key= 'ItemId'):
       
        self.k = k
        self.sample_size = sample_size
        self.weighting = weighting
        self.weighting_score = weighting_score
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        self.min_time = -1
        
        self.sim_time = 0
        
    def fit(self, train, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        self.items_ids = list(train[self.item_key].unique())
        train[self.item_key] = train[self.item_key].astype('category')
        self.new_old = dict(enumerate(train[self.item_key].cat.categories))
        self.old_new = {y:x for x,y in self.new_old.items()}
        train[[self.item_key]] = train[[self.item_key]].apply(lambda x: x.cat.codes)
        
        self.freqs = dict(train[self.item_key].value_counts())
        
        self.num_items = train[self.item_key].max()
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        session = -1
        session_items = set()
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                session = row[index_session]
                session_items = set()
            session_items.add(row[index_item])
            
            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.predict_for_item_ids = list(range(1, self.num_items+1))
        
        
    def predict_next(self, session_items, k):
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
        neighbors = self.find_neighbors(input_item_id, session_items)
        scores = self.score_items(neighbors, session_items)
        
        # Create things in the format ..
        preds = np.zeros(all_len)
        scores_keys = list(scores.keys())
        for i in range(all_len):
            if i+1 in scores_keys:
                preds[i] = scores[i+1]
                
        series = pd.Series(data = preds, index = self.predict_for_item_ids)
        series = series / series.max()
        return series.nlargest(k).index.values
    
    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    def vec_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_vec_map.get(session);
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id ) if item_id in self.item_session_map else set()
        
        
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, input_item_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id )
               
        if self.sample_size == 0: #use all session as possible neighbors
            return self.relevant_sessions

        else: #sample some sessions
            if len(self.relevant_sessions) > self.sample_size:    
                return self.relevant_sessions[-self.sample_size:]
            else: 
                return self.relevant_sessions
                        
    def calc_similarity(self, session_items, sessions):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        pos_map = {}
        length = len(session_items)
        
        count = 1
        for item in session_items:
            if self.weighting is not None: 
                pos_map[item] = getattr(self, self.weighting)(count, length)
                count += 1
            else:
                pos_map[item] = 1
        #print('POS MAP: ', pos_map, session_items)
        items = set(session_items)
        neighbors = []
        for session in sessions: 
            n_items = self.items_for_session(session)
            similarity = self.vec(items, n_items, pos_map)        
            if similarity > 0:
                neighbors.append((session, similarity))
        return neighbors

    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, input_item_id, session_items):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: list of item ids in current session
        input_item_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        #print('SESSION ITEMS1:', session_items)
        possible_neighbors = self.possible_neighbor_sessions(input_item_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)
        
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
            
    def score_items(self, neighbors, current_session):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )
            step = 1
            
            for item in reversed( current_session ):
                if item in items:
                    decay = getattr(self, self.weighting_score)(step)
                    break
                step += 1
                                    
            for item in items:
                old_score = scores.get( item )
                similarity = session[1]
                
                if old_score is None:
                    scores.update({item : ( similarity * decay ) })
                else: 
                    new_score = old_score + ( similarity * decay )
                    scores.update({item : new_score})
                    
        return scores
    
    
    def linear_score(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same_score(self, i):
        return 1
    
    def div_score(self, i):
        return 1/i
    
    def log_score(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic_score(self, i):
        return 1/(i*i)
    
    def linear(self, i, length):
        return 1 - (0.1*(length-i)) if i <= 10 else 0
    
    def same(self, i, length):
        return 1
    
    def div(self, i, length):
        return i/length
    
    def log(self, i, length):
        return 1/(log10((length-i)+1.7))
    
    def quadratic(self, i, length):
        return (i/length)**2


    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union
        
        self.sim_time += (time.clock() - sc)
        
        return res 
    
    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result
    
    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result
    
    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)
        
        result = (2 * a) / ((2 * a) + b + c)

        return result
    
    def vec(self, first, second, map):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]
        
        result = sum / len(map)

        return result    