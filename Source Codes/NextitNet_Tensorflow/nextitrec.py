import tensorflow as tf
import data_loader_recsys
import generator_recsys
import utils
import time
import numpy as np
import argparse
import os
import gc
import pandas as pd

def generatesubsequence(train_set):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        seq = train_set[i]
        lenseq = len(seq)
        for j in range(lenseq - 2):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(42)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print ("generating subsessions is done!")
    return x_train

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5, help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9, help='hyperpara-Adam')
    parser.add_argument('--datapath', type=str, default='Data/Session/user-filter-20000items-session5.csv', help='data path')
    parser.add_argument('--eval_iter', type=int, default=1000, help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=1000, help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2, help='0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False, help='whether generating a subsessions')
    args = parser.parse_args()
    exps = pd.read_csv('exp.csv')
    cPid = os.getpid()
    train_time = 0
    test_time = 0
    for i,row in exps.iterrows():
        gc.collect()
        args.expname = row['name']
        args.sessionid = row['sessionid']
        args.itemid = row['itemid']
        args.data_folder = row['path']
        args.valid_data = row['test']
        args.train_data = row['train']
        args.freq = row['freq']
        args.model_type = 'generator'
        print(("\n\n############################################\n"), args.train_data, ' --- ', args.valid_data)
        with open("LOGGER_"+ args.expname + ".txt", "a") as myfile:
            myfile.write(row['train'] + ", " + row['test'] +"\n")
            
        train_data = os.path.join(args.data_folder, args.train_data)
        args.dir_name = train_data
        dl = data_loader_recsys.Data_Loader(vars(args))
        train_set = dl.item
        items = dl.item_dict
        print ("len(train items)", len(items))
        
        valid_data = os.path.join(args.data_folder, args.valid_data)
        args.dir_name = valid_data
        vdl = data_loader_recsys.Data_Loader(vars(args), testFlag = True, itemsIDs = dl.itemsIDs, max_doc = dl.max_document_length, vocab_proc = dl.vocab_processor)
        valid_set = vdl.item
        items2 = vdl.item_dict
        print ("len(valid items)", len(items2))    
        model_para = {
            #if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
            'item_size': len(items),
            'dilated_channels': 100,#larger is better until 512 or 1024
            # if you use nextitnet_residual_block, you can use [1, 4, 1, 4, 1,4,],
            # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
            # when you change it do not forget to change it in nextitrec_generate.py
            'dilations': [1, 2, 4, ],#YOU should tune this hyper-parameter, refer to the paper.
            'kernel_size': 3,
            'learning_rate':0.001,#YOU should tune this hyper-parameter
            'batch_size':32,#YOU should tune this hyper-parameter
            'epochs':10,# if your dataset is small, suggest adding regularization to prevent overfitting
            'is_negsample':False #False denotes no negative sampling
        }
        tf.compat.v1.reset_default_graph()
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            itemrec = generator_recsys.NextItNet_Decoder(model_para)
            itemrec.train_graph(model_para['is_negsample'])
            optimizer = tf.compat.v1.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
            itemrec.predict_graph(model_para['is_negsample'],reuse=True)
            sess= tf.Session()
            init=tf.global_variables_initializer()
            sess.run(init)
        for e in range(model_para['epochs']):
            print("\n############################\nEPOCH #:", e)
            batch_no = 0
            batch_size = model_para['batch_size']
            losses = []
            t1 = time.time()
            while (batch_no + 1) * batch_size < train_set.shape[0]:
                batch_no += 1
                item_batch = train_set[(batch_no-1) * batch_size: (batch_no) * batch_size, :]
                _, loss, results = sess.run([optimizer, itemrec.loss,
                     itemrec.arg_max_prediction],feed_dict={itemrec.itemseq_input: item_batch})
                losses.append(loss)
                if batch_no % 100 == 0:
                    print('Finished Batch:', batch_no)
                    
            print('Train Loss:', np.mean(losses), valid_set.shape[0])
            train_time += (time.time() - t1)
            
            batch_no_test = 0
            batch_size_test = batch_size * 1
            MRR = [[], [], [], [], []]
            Rec = [[], [], [], [], []]
            cov = [[], [], [], [], []]
            pop = [[], [], [], [], []]
            Ks = [1, 3, 5, 10, 20]
            t1 = time.time()
            while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                batch_no_test += 1
                item_batch = valid_set[(batch_no_test-1) * batch_size_test: (batch_no_test) * batch_size_test, :]
                [probs] = sess.run([itemrec.g_probs], feed_dict={itemrec.input_predict:item_batch})
                for bi in range(probs.shape[0]):
                    true_item = item_batch[bi][-1]
                    if true_item == 1:
                        continue
                    if args.freq != 0 and true_item != 0 and dl.freqs[true_item] > args.freq:
                        continue
                    for k in range(len(Ks)):
                        pred_items = utils.sample_top_k(probs[bi][-1], top_k=Ks[k])
                        predictmap = {ch : i for i, ch in enumerate(pred_items)}
                        print(pred_items, predictmap)
                        for p in pred_items:
                            if p == 1:
                                continue
                            if p not in cov[k]:
                                cov[k].append(p)
                            pop[k].append(dl.freqs[p])
                        rank = predictmap.get(true_item)
                        if rank == None:
                            mrr = 0.0
                            rec = 0.0
                        else:
                            mrr = 1.0/(rank+1)
                            rec = 1.0
                        MRR[k].append(mrr)
                        Rec[k].append(rec)
            test_time += (time.time() - t1) / len(Ks)            
            Rec[:] = [np.mean(x) for x in Rec]
            MRR[:] = [np.mean(x) for x in MRR]
            cov[:] = [len(x) / len(items) for x in cov]
            maxi = max(dl.freqs.values())
            pop[:] = [np.mean(x) / maxi for x in pop]
            print("MRR@20:", MRR[-1])
            print("Recall@20:", Rec[-1])
            print("Cov@20:", cov[-1])
            print("Pop@20:", pop[-1])
            with open("LOGGER_"+ args.expname + ".txt", "a") as myfile:
                myfile.write('EPOCH #:' + str(e))
                myfile.write(str(Rec[0])+','+str(Rec[1])+','+str(Rec[2])+','+str(Rec[3])+','+str(Rec[4])+','+
                             str(MRR[0])+','+str(MRR[1])+','+str(MRR[2])+','+str(MRR[3])+','+str(MRR[4]))
                myfile.write("\nCOV:"+str(cov[0])+','+str(cov[1])+','+str(cov[2])+','+str(cov[3])+','+str(cov[4]))
                myfile.write("\nPOP:"+str(pop[0])+','+str(pop[1])+','+str(pop[2])+','+str(pop[3])+','+str(pop[4]))
                myfile.write("\nTrainTime:"+str(train_time))
                myfile.write("\nTestTime:"+str(test_time))
                myfile.write("\n############################################\n")
if __name__ == '__main__':
    main()