# (Item2vec) Recommendation-based-on-sequence.
 
Item2Vec uses the skip-gram with negative sampling neural word embedding to find out vector representations for different items that infers the relation between an item to its surrounding items in a session. During the prediction phase, candidate items get scores according to the similarity distance between their embedding vectors, and the average of session items embedding vectors. Source [Repo](https://github.com/Bekyilma/Recommendation-based-on-sequence-)

# Quick install

```--pip install --upgrade gensim```

# Dependencies and Requirements

Gensim is known to run on Linux, Windows and Mac OS X and should run on any other platform that supports Python 2.6+ and NumPy. <br>
Gensim depends on the following software: <br>
Python >= 2.6. <br>
NumPy >= 1.3. <br>
SciPy >= 0.7. <br>

## Usage

### Dataset
- Contents
        - A column in the file should be the integer Session IDs with header name SessionID
        - A column in the file should be the integer Item IDs with header name ItemID

### List of Arguments accepted

```--n_epochs, default=3, type=int```,  #number of epochs <br>
```--embedding_dim, type=int, default=15```, #Embedding Vector dimension <br>
```--sg, type=int, default=1```, "Use SkipGram 1 or CBOW 0" <br>
```--window, type=int, default=3```, #Window Size <br>
```--min_threshold, type=int, default=5```, #Minimum word frequency threshold <br>
```--negative, type=int, default=20```, #Negative Samples to be used <br>
```--initial_lr, default=0.025, type=float```, #Initial Learning Rate <br>
```--final_lr, default=1e-4, type=float```, #Final Learning Rate <br>
```--sample, default=1e-4, type=float```, #Subsampling ratio of most frequent words <br>
```--seed, type=int, default=22```, #Random Seed <br>
```--K, type=int, default=20``` #K items to be used in Recall@K and MRR@K <br>
```--valid_data, default='DataValid.csv', type=str ``` #name of Validation split file <br>
```--train_data, default='DataTrain.csv', type=str ``` #name of training split file <br>
```--data_folder, default='', type=str``` #directory to the main data folder 

## Results
- All the results for item2vec can be seen from [HERE](https://github.com/mmaher22/iCV-SBR/blob/master/Results/Item2Vec.pdf).
