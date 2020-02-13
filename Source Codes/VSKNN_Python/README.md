# Vector multiplication session-based K-nearest neighbors (VSKNN)
- Python Implementation of the algorithm.
- Original paper: VSKNN can be found => [Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/pdf/1803.09587.pdf)
- This code is based on [Evaluation of Session-based Recommendation Algorithms](https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AACldzQWbw-igKjcPTBI6ZPAa?dl=0) but adapted to our experiments.

## Requirements
- Python 3.x
- pandas
- numpy

## Usage
### Dataset
- requirements:
        - A column in the file should be the integer Session IDs with header name SessionID
        - Several columns of the file should be the item features and must include Item IDs
        
### Contents
The project have a structure as below:

```bash
├── VSKNN_Python
│   ├── vsknn.py
│   ├── main_vsknn.py
```
`vsknn.py` is the file for the corresponding algorithm class with fit and predict methods.
`main_vsknn.py` is the file containing the logic for defining arguments used, reading datasets, fitting models, predicting and evaluating the algorithms.

Example
```bash
python main_vsknn.py --data_folder Dataset/ --train_data train.csv --valid_data valid.csv --K 20 --sample 2000 --itemid ItemID --sessionid sessionID
```

### List of Arguments accepted for Session-based Matrix Factorization(SMF)
```--factors``` type = int, default = 100, help = Number of latent factors. <br>
```--K``` type = int, default = 20, help = K items to be used in Recall@K and MRR@K. <br>
parser.add_argument('--neighbors', type=int, default=200, help="K neighbors to be used in KNN")
```--sample```, type=int, default=0, help=Max Number of steps to walk back from the currently viewed item. <br>
```--weight_score```, type=str, default='div_score', help=Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic) score. <br>
```--weighting```, type=str, default='div', help=Decay function to determine the importance/weight of individual actions in the current session(linear, same, div, log, qudratic). <br>
```--similarity```, type=str, default='cosine', help = String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). <br>
```--itemid```, type=str, default='ItemID', help = name of ItemID column in dataset files. <br>
```--sessionid```, default='SessionID', help = name of sessionID column in dataset files. <br>
```--item_feats```, default='', help = Names of Columns containing items features separated by #. <br>
```--valid_data```, default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data```, default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder```, help=directory containing dataset splits. 

## Results

- All the results for SRGNN can be seen from [HERE](https://github.com/mmaher22/iCV-SBR/blob/master/Results/VSKNN.pdf).
