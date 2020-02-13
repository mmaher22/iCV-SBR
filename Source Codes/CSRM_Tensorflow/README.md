# Session-based Matrix Factorization(SMF)
- Session-based Matrix Factorization for Session Based Recommendation.
- Python Implementation of the algorithms.
- Original paper: SMF can be found => [Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/pdf/1803.09587.pdf)
- This code is based on [Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/pdf/1803.09587).

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
        
### Training and Testing
The project have a structure as below:

```bash
├── smf_py
│   ├── smf.py
│   ├── smf_main.py
```
`smf.py` is the file for the corresponding algorithm class with fit and predict methods.
`smf_main.py` is the file containing the logic for defining arguments used, reading datasets, fitting models, predicting and evaluating the algorithms.

Example
```bash
python smf_main.py --data_folder Dataset/ --train_data train.csv --valid_data valid.csv --K 20  --itemid ItemID --sessionid sessionID
```

### List of Arguments accepted for Session-based Matrix Factorization(SMF)
```--factors``` type = int, default = 100, help = Number of latent factors. <br>
```--K``` type = int, default = 20, help = K items to be used in Recall@K and MRR@K. <br>
```--epochs``` default=10, help=Number of Epochs. <br>
```--batch```, type=int, default=32, help=Batch size for the training process. <br>
```--momentum```, type=float, default=0.0, help=Momentum of the optimizer adagrad_sub. <br>
```--regularization```, type=float, default=0.0001, help=Regularization Amount of the objective function. <br>
```--dropout```, type=float, default=0.0, help=Share of items that are randomly discarded from the current session while training. <br>
```--skip```, type=float, default=0.0, help=Probability that an item is skiped and the next one is used as the positive example. <br>
```--neg_samples```, type=int, default=2048, help=Number of items that are sampled as negative examples. <br>
```--activation```, type=str, default=linear, help=Final activation function (linear, sigmoid, uf_sigmoid, hard_sigmoid, relu, softmax, softsign, softplus, tanh). <br>
```--objective```, type=str, default=bpr_max, help=Loss Function (bpr_max, top1_max, bpr, top1). <br>
```--lr```, type=float, default=0.001, help=Learning Rate. <br>
```--itemid```, type=str, default='ItemID', help = name of ItemID column in dataset files. <br>
```--sessionid```, default='SessionID', help = name of sessionID column in dataset files. <br>
```--item_feats```, default='', help = Names of Columns containing items features separated by #. <br>
```--valid_data```, default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data```, default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder```, help=directory containing dataset splits. 



## Results

- Different different parameters have been tried out using smf on a sample of RecSys15 Challenge Dataset.
- All the results can be seen from [HERE](https://docs.google.com/spreadsheets/d/19z6zFEY6pC0msi3wOQLk_kJsvqF8xnGOJPUGhQ36-wI/edit#gid=0).