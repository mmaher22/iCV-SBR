# Improved GRU4Rec
- Recurrent Neural Networks with Top-k Gains for Session-based Recommendations.
- Theano Implementation of the model.
- Original paper: => [Recurrent Neural Networks with Top-k Gains for Session-based Recommendations(CIKM 2018)](https://arxiv.org/abs/1706.03847).
- This code is based on [Original Github Repository](https://github.com/hidasib/GRU4Rec) but adapted according to our experiments.

## Requirements
- Python 3.x
- pandas
- numpy
- Theano 1.0.0

## Usage

### Dataset tested
RecSys Challenge 2015 Dataset can be retreived from [HERE](https://2015.recsyschallenge.com/).

### Pre processing data

The format of data is similar to that obtained from RecSys Challenge 2015:
- Filenames
    - Training set by default is named as `recSys15TrainOnly.txt`
    - Test set by default is named as `recSys15Valid.txt`
- Contents
    - `recSys15TrainOnly.txt`, `recSys15Valid.txt` should be the csv files that stores the pandas dataframes that satisfy the following requirements:
        - A column in the file should be the integer Session IDs with header name SessionID
        - Several columns of the file should be the item features and must include Item IDs
        - The 3rd column of the file should be the Timestamps with header name Time
        
### Training and Testing
The project have a structure as below:

```bash
├── main_gru4rec.py
├── GRU4Rec
│   ├── evaluation.py
│   ├── gru4rec.py
```
`gru4rec.py` is file for corrsponding model definition, training, and testing. <br>
`evaluation.py` is file for evaluation of already trained model.<br>
`main_gru4rec.py` file is used for the corrsponding logic for defining arguments used, reading datasets, preprocessing, and calling model.

Example
```bash
python main_gru4rec.py --data_folder Dataset/ --train_data train.csv --valid_data valid.csv --K 20 --itemid ItemID --sessionid SessionID --timekey Time --gpu True
```

### List of Arguments accepted for Sequential Rules

```--K``` type=int, default=20, help=K items to be used in Recall@K and MRR@K <br>
```--timekey``` default='Time', type=str, help=header of the timestamp column in the input file. <br>
```--itemid``` default='ItemID', type=str, help=header of the ItemID column in the input file. <br>
```--sessionid``` default='SessionID', type=str, help=header of the SessionID column in the input file. <br>
```--valid_data``` default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data``` default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder``` help=directory containing dataset splits.<br>
```--optimizer``` default='adagrad', type=str, help=sets the appropriate learning rate adaptation strategy, use None for standard SGD (None, default:'adagrad', 'rmsprop', 'adam', 'adadelta') <br>
```--neg_sample``` type=int, default=2048, help=number of additional negative samples to be used (besides the other examples of the minibatch. <br>
```--batch_size``` type=int, default=32, help=size of the minibacth, also effect the number of negative samples through minibatch based sampling. <br>
```--n_epoch``` type=int, default=10, help=number of training epochs. <br>
```--n_hidden``` type=int, default=100, help=list of the number of GRU units in the layers. <br>
```--embedding``` type=int, default=10, help=size of the embedding used, 0 means not to use embedding. <br>
```--loss``` type=str, default='bpr-max', help=selects the loss function ('top1', 'bpr', 'cross-entropy', 'xe_logit', 'top1-max', 'bpr-max'). <br>
```--lr``` type=float, default=0.05, help=learning rate. <br>
```--act``` type=str, default='tanh', help='linear', 'relu', default:'tanh', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>' selects the activation function on the hidden states, <X> and <Y> are the parameters of the activation function. <br>
```--final_act``` type=str, default='elu-0.5', help='linear', 'relu', 'tanh', 'leaky-<X>', default:'elu-0.5', 'selu-<X>-<Y>' selects the activation function on the final layer, <X> and <Y> are the parameters of the activation function. <br>
```--dropout``` type=float, default=0.0, help=probability of dropout of hidden units. <br>
```--momentum``` type=float, default=0.1, help=if not zero, Nesterov momentum will be applied during training with the given strength. <br>
```--bpreg``` type=float, default=1.0, help=score regularization coefficient for the BPR-max loss function. <br>
```--constrained_embedding``` type=bool, default=False, help=if True, the output weight matrix is also used as input embedding. <br>
```--gpu``` type=bool, default=False, help=Either to train using GPU or not.

## Results

- Different different parameters have been tried out on samples from RecSys15 Challenge/Diginetica Datasets.
- All the results can be seen from [HERE](https://docs.google.com/spreadsheets/d/19z6zFEY6pC0msi3wOQLk_kJsvqF8xnGOJPUGhQ36-wI/edit#gid=0).
