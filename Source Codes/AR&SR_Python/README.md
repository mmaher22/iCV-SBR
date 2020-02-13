# AR-SR-recommendation
- Association Rules and Sequential Rules for Session Based Recommendation supporting multi-item features.
- Python Implementation of the algorithms.
- Original paper: Association Rules => [Mining Association Rules Between Sets of Items in Large Databases(SIGMOD 1993)](https://rakesh.agrawal-family.com/papers/sigmod93assoc.pdf) and Sequential Rules => [A Comparison of Frequent Pattern Techniques and a Deep Learning Method for Session-Based Recommendation(RecSys 2017)](http://ceur-ws.org/Vol-1922/paper10.pdf)
- This code is based on [Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/pdf/1803.09587) but adapted to support multi-item features other than ItemID.

## Requirements
- Python 3.x
- pandas
- numpy

## Usage

### Dataset
RecSys Challenge 2015 Dataset can be retreived from [HERE](https://2015.recsyschallenge.com/)

### Pre processing data
- The training set is divided into training and testing where the testing split is the last day sessions.

The format of data is similar to that obtained from RecSys Challenge 2015:
- Filenames
    - Training set should be named as `recSys15TrainOnly.txt`
    - Test set should be named as `recSys15Valid.txt`
- Contents
    - `recSys15TrainOnly.txt`, `recSys15Valid.txt` should be the csv files that stores the pandas dataframes that satisfy the following requirements:
        - A column in the file should be the integer Session IDs with header name SessionID
        - Several columns of the file should be the item features and must include Item IDs
        - The 3rd column of the file should be the Timestamps with header name Time
        
### Training and Testing
The project have a structure as below:

```bash
├── AssociationRules
│   ├── AR.py
│   ├── mainAR.py
├── SequentialRules
│   ├── SR.py
│   ├── mainSR.py
```
`SR.py` and `AR.py` are files for corrsponding classes of Sequential Rules and Association Rules algorithms respectively.
`mainSR.py` and `mainAR.py` are files for corrsponding logic for defining arguments used, reading datasets, fitting models, predicting and evaluating the algorithms.

Example
```bash
python mainAR.py --data_folder Dataset/ --train_data train.csv --valid_data valid.csv --K 20 --prune 0 --itemid ItemID --sessionid sessionID
```

### List of Arguments accepted for Sequential Rules
```--prune``` type = int, default = 0, help = Association Rules Pruning Parameter. <br>
```--K``` type = int, default = 20, help = K items to be used in Recall@K and MRR@K. <br>
```--steps``` type=int, default=10, help = Max Number of steps to walk back from the currently viewed item. <br>
```--weighting``` type=str, default='div', help = Weighting function for the previous items (linear, same, div, log, qudratic). <br>
```--itemid``` default='ItemID', help = name of ItemID column in dataset files. <br>
```--sessionid``` default='SessionID', help = name of sessionID column in dataset files. <br>
```--item_feats``` default='', help = Names of Columns containing items features separated by #. <br>
```--valid_data``` default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data``` default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder``` help=directory containing dataset splits.

### List of Arguments accepted for Association Rules
```--prune``` type = int, default = 0, help = Association Rules Pruning Parameter. <br>
```--K``` type = int, default = 20, help = K items to be used in Recall@K and MRR@K. <br>
```--itemid``` default='ItemID', help = name of ItemID column in dataset files. <br>
```--sessionid``` default='SessionID', help = name of sessionID column in dataset files. <br>
```--item_feats``` default='', help = Names of Columns containing items features separated by #. <br>
```--valid_data``` default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data``` default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder``` help=directory containing dataset splits.

## Results

- Different different parameters have been tried out using both models on RecSys15 Challenge Dataset.
- Best AR results were 53.44%,	21.31% for Recall@20 and MRR@20 respectively.
- Best SR results were 53.81%, 21.61% for Recall@20 and MRR@20 respectively.
- All the results can be seen from [HERE](https://docs.google.com/spreadsheets/d/19z6zFEY6pC0msi3wOQLk_kJsvqF8xnGOJPUGhQ36-wI/edit#gid=0).
