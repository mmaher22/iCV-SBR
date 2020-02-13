# S-POP-recommendation
- Session popularity predictor that gives higher scores to items with higher number of occurrences in the session. Ties are broken up by adding the popularity score of the item.
- Python Implementation of the algorithms.
- This code is based on [Evaluation of Session-based Recommendation Algorithms](https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AACldzQWbw-igKjcPTBI6ZPAa?dl=0), [PAPER](https://arxiv.org/pdf/1803.09587) but adapted to support multi-item features other than ItemID and other requirements of our experiments.
- The score is given by:
```math
r_{s,i} = supp_{s,i} + \frac{supp_i}{(1+supp_i)}
```
## Requirements
- Python 3.x
- pandas
- numpy

## Usage

### Dataset
- Contents
        - A column in the file should be the integer Session IDs with header name SessionID
        - Several columns of the file should be the item features and must include Item IDs
        
### Training and Testing
The project have a structure as below:

```bash
├── main_spop.py
├── spop.py
```
`spop.py` is the file of corrsponding class of S-POP algorithm.
`main_spop.py` is the file of corrsponding logic for defining arguments used, reading datasets, fitting models, predicting and evaluating the algorithms.

Example
```bash
python main_spop.py --data_folder Dataset/ --train_data train.csv --valid_data valid.csv --K 20 --topn 100 --itemid ItemID --sessionid sessionID
```

### List of Arguments accepted for S-POP
```--topn``` type = int, default = 100, help = Only give back non-zero scores to the top N ranking items. <br>
```--K``` type = int, default = 20, help = K items to be used in Recall@K and MRR@K. <br>
```--itemid``` default='ItemID', help = name of ItemID column in dataset files. <br>
```--sessionid``` default='SessionID', help = name of sessionID column in dataset files. <br>
```--valid_data``` default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data``` default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder``` help=directory containing dataset splits.

## Results

- All the results can be seen from [HERE](https://github.com/mmaher22/iCV-SBR/blob/master/Results/S-POP.pdf).
