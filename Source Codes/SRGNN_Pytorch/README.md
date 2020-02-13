# SRGNN (Graph Neural Networks)
- Graph Neural Networks in Session based Recommendation supporting multi-item features.
- Pytorch Implementation of the model.
- Original paper: SRGNN => [Session-based Recommendation with Graph Neural Networks(AAAI 2019)](https://sxkdz.github.io/research/SR-GNN/).
- This code is based on [Original Github Repository](https://github.com/CRIPAC-DIG/SR-GNN) but adapted to support multi-item features other than ItemID.

## Requirements
- Python 3.x
- pandas
- numpy
- Pytorch 0.4.0

## Usage

### Dataset
- requirements:
        - A column in the file should be the integer Session IDs with header name SessionID
        - Several columns of the file should be the item features and must include Item IDs
        - The 3rd column of the file should be the Timestamps with header name Time
        
### Contents
The project have a structure as below:

```bash
├── SRGNN
│   ├── utils.py
│   ├── model.py
│   ├── main.py
```
`model.py` is file for corrsponding model definition, training, and testing. `utils.py` is used for data reader and batching / slicing. 
`main.py` file is used for the corrsponding logic for defining arguments used, reading datasets, preprocessing, and calling model.

Example
```bash
python main.py --data_folder Dataset/ --train_data train.csv --valid_data valid.csv --K 20 --itemid ItemID --sessionid sessionID
```

### List of Arguments accepted for Sequential Rules
```--K``` type = int, default = 20, help = K items to be used in Recall@K and MRR@K. <br>
```--itemid``` default='ItemID', help = name of ItemID column in dataset files. <br>
```--sessionid``` default='SessionID', help = name of sessionID column in dataset files. <br>
```--item_feats``` default='', help = Names of Columns containing items features separated by #. <br>
```--valid_data``` default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data``` default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder``` help=directory containing dataset splits.<br>
```--batchSize``` default=100, help = input batch size. <br>
```'--hiddenSize``` default=100, help = hidden state size. <br>
```--epoch``` default=30, help = the number of epochs to train for. <br>
```--lr``` default=0.001, help = learning rate. <br>
```--lr_dc``` default=0.1, help = 'learning rate decay rate. <br>
```--lr_dc_step``` default=3, help = the number of steps after which the learning rate decay. <br>
```--l2``` default=1e-5, help = l2 penalty. <br>
```--step``` default=1, help = gnn propogation steps.


## Results

- All the results for AR can be seen from [HERE](https://github.com/mmaher22/iCV-SBR/blob/master/Results/SRGNN.pdf).
