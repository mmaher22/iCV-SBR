import os
import time
import argparse
import pandas as pd
from GRU4REC.gru4rec import GRU4Rec
from GRU4REC.evaluation import evaluate_sessions_batch, evaluate_gpu

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--timekey', default='Time', type=str, help="header of the timestamp column in the input file (default: 'Time')")
parser.add_argument('--itemid', default='ItemID', type=str, help="header of the ItemID column in the input file (default: 'ItemID')")
parser.add_argument('--sessionid', default='SessionID', type=str, help="header of the SessionID column in the input file (default: 'SessionID')")
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--data_folder', default='../Dataset/RecSys_Dataset_After/', type=str)

parser.add_argument('--optimizer', default='adagrad', type=str, help="sets the appropriate learning rate adaptation strategy, use None for standard SGD (None, default:'adagrad', 'rmsprop', 'adam', 'adadelta')")
parser.add_argument('--neg_sample', type=int, default=2048, help="number of additional negative samples to be used (besides the other examples of the minibatch) (default: 2048)")
parser.add_argument('--batch_size', type=int, default=32, help="size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 32)")
parser.add_argument('--n_epoch', type=int, default=10, help="number of training epochs (default: 10)")
parser.add_argument('--embedding', type=int, default=10, help="size of the embedding used, 0 means not to use embedding (default: 0)")
parser.add_argument('--n_hidden', type=int, default=100, help="list of the number of GRU units in the layers (default : [100])")
parser.add_argument('--loss', type=str, default='bpr-max', help="selects the loss function ('top1', 'bpr', 'cross-entropy', 'xe_logit', 'top1-max', default:'bpr-max')")
parser.add_argument('--lr', type=float, default=0.05, help="learning rate (default: 0.05)")
parser.add_argument('--act', type=str, default='tanh', help="'linear', 'relu', default:'tanh', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>' selects the activation function on the hidden states, <X> and <Y> are the parameters of the activation function")
parser.add_argument('--final_act', type=str, default='elu-0.5', help="'linear', 'relu', 'tanh', 'leaky-<X>', default:'elu-0.5', 'selu-<X>-<Y>' selects the activation function on the final layer, <X> and <Y> are the parameters of the activation function")
parser.add_argument('--dropout', type=float, default=0.0, help="probability of dropout of hidden units (default: 0.0)")
parser.add_argument('--momentum', type=float, default=0.1, help="if not zero, Nesterov momentum will be applied during training with the given strength (default: 0.1)")
parser.add_argument('--bpreg', type=float, default=1.0, help="score regularization coefficient for the BPR-max loss function (default: 1.0)")
parser.add_argument('--constrained_embedding', type=bool, default=False, help="if True, the output weight matrix is also used as input embedding (default: False)")
parser.add_argument('--gpu', type=bool, default=False, help="Either to train using GPU or not (default: False)")

# Get the arguments
args = parser.parse_args()
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)

print('STARTING TRAINING:')
model = GRU4Rec(loss=args.loss, final_act=args.final_act, hidden_act=args.act, layers=[args.n_hidden],
                 momentum=args.momentum, bpreg=args.bpreg, n_epochs=args.n_epoch, batch_size=args.batch_size, 
                 dropout_p_hidden=args.dropout, learning_rate=args.lr, embedding=args.embedding, 
                 n_sample=args.neg_sample, constrained_embedding=args.constrained_embedding, 
				 adapt=args.optimizer, session_key=args.sessionid, item_key=args.itemid, time_key=args.timekey)

t1 = time.time()
model.fit(x_train)
t2 = time.time()
print('FINISHED TRAINING IN:', t2 - t1)

if args.gpu == False:
    recall, mrr = evaluate_sessions_batch(model, x_valid, cut_off=args.K, batch_size=args.batch_size, 
                                      session_key=args.sessionid, item_key=args.itemid, time_key=args.timekey) 
else:
    recall, mrr = evaluate_gpu(model, x_valid, cut_off=args.K, batch_size=args.batch_size, 
                                      session_key=args.sessionid, item_key=args.itemid, time_key=args.timekey)
t3 = time.time()
print('FINISHED EVALUATION IN:', t3 - t2)
print('Recall:', recall, ' --- MRR:', mrr)