import pandas as pd
import numpy as np
from tensorflow.contrib import learn
from tflearn.data_utils import pad_sequences

def load_sequence(from_path, itemid, sessionid, Train = True, itemsIDs = []):
    data = pd.read_csv(from_path)
    if Train == True:
        itemsIDs = list(data[itemid].unique())
        for i in range(len(itemsIDs)):
            itemsIDs[i] += 1
        
    patterns = []
    cnt_session = -1
    cnt_pattern = []
    for i in range(len(data)):
        if i % 100000 == 0:
            print('Finished Till Now: ', i)
        sid = data.loc[i, [sessionid]][0]
        iid = data.loc[i, [itemid]][0] + 1
        if sid != cnt_session:
            cnt_session = sid
            cnt_pattern = []
        if Train == False and iid not in itemsIDs:
            continue
		if len(cnt_pattern) > 0 and iid == cnt_pattern[-1]:
            continue
        cnt_pattern.append(iid)
        if len(cnt_pattern) > 1:
            lst_pattern = []
            if len(patterns) > 0:
                lst_pattern = patterns[-1]
            if cnt_pattern != lst_pattern:
                patterns.append(cnt_pattern)
    
    return patterns, itemsIDs

# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options, testFlag = False, itemsIDs = [], max_doc = 0, vocab_proc = None):
        patterns, itemsIDs = load_sequence(options['dir_name'], options['itemid'], 
                                          options['sessionid'], Train = not testFlag, 
                                          itemsIDs = itemsIDs)
        self.itemsIDs = itemsIDs
        if testFlag == False:
            self.max_document_length = max([len(x) for x in patterns])
        else:
            self.max_document_length = max_doc
        patterns = pad_sequences(patterns, maxlen=self.max_document_length, padding='pre')
        positive_examples2 = [','.join(str(i) for i in x) for x in patterns]
        if testFlag == False:
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length)
        else:
            self.vocab_processor = vocab_proc
        self.item = np.array(list(self.vocab_processor.fit_transform(positive_examples2)))
        un, un_freq = np.unique(self.item, return_counts = True)
        print(un)
        print(un_freq)
        self.freqs = dict(zip(un[1:], un_freq[1:]))
        self.item_dict = self.vocab_processor.vocabulary_._mapping

    def load_generator_data(self, sample_size):
        text = self.text
        mod_size = len(text) - len(text)%sample_size
        text = text[0:mod_size]
        text = text.reshape(-1, sample_size)
        return text, self.vocab_indexed


    def string_to_indices(self, sentence, vocab):
        indices = [ vocab[s] for s in sentence.split(',') ]
        return indices

    def inidices_to_string(self, sentence, vocab):
        id_ch = { vocab[ch] : ch for ch in vocab } 
        sent = []
        for c in sentence:
            if id_ch[c] == 'eol':
                break
            sent += id_ch[c]
        return "".join(sent)