import os
import argparse
import logging
import numpy as np
from gensim.models import Word2Vec
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=3, type=int) #number of epochs
parser.add_argument("--embedding_dim", type=int, default=15, help="Embedding Vector dimension") 
parser.add_argument("--sg", type=int, default=1, help="Use SkipGram 1 or CBOW 0")
parser.add_argument("--window", type=int, default=3, help="Window Size")
parser.add_argument("--min_threshold", type=int, default=5, help="Minimum word frequency threshold")
parser.add_argument("--negative", type=int, default=20, help="Negative Samples to be used")
parser.add_argument('--initial_lr', default=0.025, type=float, help="Initial Learning Rate")
parser.add_argument('--final_lr', default=1e-4, type=float, help="Final Learning Rate")
parser.add_argument('--sample', default=1e-4, type=float, help="Subsampling ratio of most frequent words")
parser.add_argument('--seed', type=int, default=22, help="Random Seed") 
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K") 
parser.add_argument('--valid_data', default='DataValid.csv', type=str)
parser.add_argument('--train_data', default='DataTrain.csv', type=str)
#parser.add_argument('--train_data', default='retail_dataset.csv', type=str)
parser.add_argument('--data_folder', default='', type=str)

# Get the arguments
args = parser.parse_args()
train_data = os.path.join(args.data_folder, args.train_data)
valid_data = os.path.join(args.data_folder, args.valid_data)

def load_sequence(from_path):
    patterns = []
    with open(from_path) as fp:
        [patterns.append(line.strip().split(",")) for line in fp]
    return patterns

train_patterns = load_sequence(train_data)
test_patterns = load_sequence(valid_data)
# split patterns to train_patterns and test_patterns
#patterns = load_sequence(train_data)
#train_patterns = np.random.choice(patterns, int(len(patterns) * 0.8))
#test_patterns = np.random.choice(patterns, int(len(patterns) * 0.2))

# Word vector representation learning
model = Word2Vec(train_patterns, sg = args.sg, size = args.embedding_dim, window = args.window, 
                 min_count = args.min_threshold, iter=args.n_epochs, sample=args.sample,
                 alpha = args.initial_lr, min_alpha = args.final_lr, negative=args.negative)
# Test
test_size = float(len(test_patterns))
hit = 0.0
MRR = 0.0
for current_pattern in test_patterns:
    #Short session or no enough items
    if len(current_pattern) < 2:
        test_size -= 1.0
        continue
    # Reduce the current pattern in the test set by removing the last item
    last_item = current_pattern.pop()
    # Keep those items in the reduced current pattern, which are also in the models vocabulary
    items = [it for it in current_pattern if it in model.wv.vocab]
    if len(items) <= 2:
        test_size -= 1.0
        continue
    # Predict the most similar items to items
    prediction = model.wv.most_similar(positive = items, topn = args.K)
    # Evaluation
    rank = 0
    for predicted_item, score in prediction:
        rank += 1
        #print('Predicted item', predicted_item, score, ' -- True Item:', last_item)
        if predicted_item == last_item:
            hit += 1.0
            MRR += 1/rank

print('Recall: {}'.format(hit / test_size))
print ('\nMRR: {}'.format(MRR / test_size))