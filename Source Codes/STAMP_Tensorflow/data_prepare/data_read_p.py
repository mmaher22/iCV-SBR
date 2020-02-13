import numpy as np
import pandas as pd
from util.sample import Sample
from util.samplepack import Samplepack

def load_data_p(train_file, test_file, args, pro):
    # the global param.
    items2idx = {}  # the return
    items2idx['<pad>'] = 0
    idx_cnt = 0
    # load the data
    train_data, idx_cnt = _load_data(train_file, items2idx, idx_cnt, args, pro, pad_idx = 0)
    test_data, _ = _load_data(test_file, items2idx, idx_cnt, args, pad_idx = 0)
    item_num = len(items2idx.keys()) - 1 #number of items without padding
    return train_data, test_data, items2idx, item_num

def _load_data(file_path, item2idx, idx_cnt, args, pro = None, pad_idx=0):
    data = pd.read_csv(file_path, sep=',', dtype={args.itemid: np.int64})
    data.sort_values([args.sessionid], inplace=True)
    #use a sample of the dataset only
    if pro is not None:
        data = data.iloc[-int(len(data) / pro) :]   #just take 1/pro last instances
    session_data = list(data[args.sessionid].values)
    item_event = list(data[args.itemid].values)

    samplepack = Samplepack()
    samples = []
    now_id = 0
    sample = Sample()
    last_id = None
    click_items = []

    for s_id,item_id in zip(session_data, item_event):
        if last_id is None:
            last_id = s_id
        if s_id != last_id:
            item_dixes = []
            for item in click_items:
                if item not in item2idx:
                    if idx_cnt == pad_idx:
                        idx_cnt += 1
                    item2idx[item] = idx_cnt
                    idx_cnt += 1
                item_dixes.append(item2idx[item])
            in_dixes = item_dixes[:-1]
            out_dixes = item_dixes[1:]
            sample.id = now_id
            sample.session_id = last_id
            sample.click_items = click_items
            sample.items_idxes = item_dixes
            sample.in_idxes = in_dixes
            sample.out_idxes = out_dixes
            samples.append(sample)
            sample = Sample()
            last_id =s_id
            click_items = []
            now_id += 1
        else:
            last_id = s_id
        click_items.append(item_id)
    sample = Sample()
    item_dixes = []
    for item in click_items:
        if item not in item2idx:
            if idx_cnt == pad_idx:
                idx_cnt += 1
            item2idx[item] = idx_cnt
            idx_cnt += 1
        item_dixes.append(item2idx[item])
    in_dixes = item_dixes[:-1]
    out_dixes = item_dixes[1:]
    #append last session in the dataset
    sample.id = now_id
    sample.session_id = last_id
    sample.click_items = click_items
    sample.items_idxes = item_dixes
    sample.in_idxes = in_dixes
    sample.out_idxes = out_dixes
    samples.append(sample)
    #append all samples to the samplepack
    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack, idx_cnt


