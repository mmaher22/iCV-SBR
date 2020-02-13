# Short-Term Attention/Memory Priority Model (STAMP)
- STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation.
- Tensorflow Implementation of the model.
- Original paper: STAMP can be found => [STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](https://www.kdd.org/kdd2018/accepted-papers/view/stamp-short-term-attentionmemory-priority-model-for-session-based-recommend)
- This code is based on [Original GitHub Repository](https://github.com/uestcnlp/STAMP).

## Requirements
- Python 3.x
- Tensorflow 2.x
- pandas
- numpy

## Usage

### Dataset
- requirements:
        - A column in the file should be the integer Session IDs with header name SessionID
        - Several columns of the file should be the item features and must include Item IDs
        - The 3rd column of the file should be the Timestamps with header name Time
        
### Contents
The project have a structure as below:

```bash
├── cmain.py (Main Script)
├── basic_layer
├── data_prepare
│   ├── data_read_p.py
├── model
│   ├── STAMP_rsc.py (Model Main Class)
├── util
```
`STAMP_rsc.py` is the file for the corresponding model algorithm class with fit and predict methods.
`cmain.py` is the file containing the logic for defining arguments used, reading datasets, calling model class, orchasterate between model class functions.
`data_read_p.py` is the file corresponding to reading the dataset properly and conversion to the needed data structure.
`util` is the folder containing scripts for Data batching, sampling, saving, randomizers, performance calculation, etc.
`basic_layer` is the folder containing scripts different layer types used in the model.

Example
```bash
python cmain.py --data_folder Dataset/ --train_data train.csv --valid_data valid.csv --K 20  --itemid ItemID --sessionid sessionID
```

### List of Arguments accepted by STAMP

```--K``` type = int, default = 20, help = K items to be used in Recall@K and MRR@K. <br>
```--epoch``` default=5, help=Number of Epochs. <br>
```--sample``` default=64, help=Use last 1/sample portion of the dataset. <br>
```--batch```, type=int, default=32, help=Batch size for the training process. <br>
```--edim```, type=int, default=100, help=Embeddings Dimension. <br>
```--hidden```, type=int, default=100, help=Hidden Layer Dimension. <br>
```--max_grad_norm```, type=int, default=150, help=Maximum Gradient Norm. <br>
```--activation```, type=str, default=linear, help= activation function (sigmoid, relu, tanh). <br>
```--lr```, type=float, default=0.001, help=Learning Rate. <br>
```--stddev```, type=float, default=0.05, help=standard deviation of normal distribution used in weights initialization. <br>
```--savemodel```, type=str, default=True, help = save model or not. <br>
```--nottrain```, type=str, default=False, help = Set true if using pretrained model for testing. <br>
```--modelpath```, type=str, default='', help = model path in case of testing only. <br>
```--itemid```, type=str, default='ItemID', help = name of ItemID column in dataset files. <br>
```--sessionid```, default='SessionID', help = name of sessionID column in dataset files. <br>
```--valid_data```, default='recSys15Valid.txt', help = default validation set name. <br>
```--train_data```, default='recSys15TrainOnly.txt', help=default training set name. <br>
```--data_folder```, help=directory containing dataset splits. 


## Results

- All the results for STAMP can be seen from [HERE](https://github.com/mmaher22/iCV-SBR/blob/master/Results/SRGNN.pdf).
