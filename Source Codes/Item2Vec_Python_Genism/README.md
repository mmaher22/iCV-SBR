# (Item2vec) Recommendation-based-on-sequence.
 
Summary of the architecture and description of the algorithm can be found [HERE](https://docs.google.com/document/d/1YAiFAsXw-uLovMu9-k89shCo-SYkOUXTGfJ4vJ-zNOM)

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
- RecSys Challenge 2015 Dataset can be retreived from [HERE](https://2015.recsyschallenge.com/)
- Train Split is the whole Traning set except for the last day which is used in the validation process.
- Data needs to be preprocessed such that each line contains a session with the clicked items IDs separated by commas.

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
Different loss functions and different parameters have been tried out and the results can be seen from [HERE](https://docs.google.com/spreadsheets/d/19z6zFEY6pC0msi3wOQLk_kJsvqF8xnGOJPUGhQ36-wI/edit#gid=0)
