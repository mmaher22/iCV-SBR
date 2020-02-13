# Comprehensive Benchmarking of Session-based Recommendation Deep-Learning Approaches

<div style="text-align: center">
<img src="Figures/MLPipe-1.png" width="900px" atl="Machine Learning Pipeline"/>
</div>
In this repository, we present a comprehensive evaluation of the state-of-the-art deep learning approaches used in session-based recommendation. Our experiments compares between baseline techniques like nearest neighbors and pattern mining algorithms, and deep learning approaches including recurrent neural networks, graph neural networks, and attention based networks. It has been shown that advanced neural network models outperformed the baseline techniques in most of the scenarios however they still suffer more in case of cold start problem. This repo contains the implementations of different algorithms used in our experiments, survey of session-based recommendation papers, and a summary of the paper resources, and results.

## This work was done in <a href = "http://icv.tuit.ut.ee/">iCV Lab</a>, University of Tartu:
<div style="text-align: center">
<a href = "http://icv.tuit.ut.ee/"><img src="http://icv.tuit.ut.ee/wp-content/uploads/2018/10/Layer-1-1.png" width="400px" atl="iCV Lab University of Tartu"/> </a>
</div>

## Funded by Rakuten , Inc. (grant VLTTI19503):
<div style="text-align: center">
 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Rakuten_Global_Brand_Logo.svg/1200px-Rakuten_Global_Brand_Logo.svg.png" width="300px" atl="iCV Lab University of Tartu"/>
</div>

<hr>

## Table of Contents & Organization:
This repository will be organized into 6 separate sections:
+ [Meta-Learning Techniques for AutoML search problem](#meta-learning-techniques-for-automl-search-problem)
  - [Learning From Model Evaluation](#learning-from-model-evaluation)
    - [Surrogate Models](#surrogate-models)
    - [Warm-Started Multi-task Learning](#warm-started-multi-task-learning)
    - [Relative Landmarks](#relative-landmarks)
  - [Learning From Task Properties](#learning-from-task-properties)
    - [Using Meta-Features](#using-meta-features)
    - [Using Meta-Models](#using-meta-models)
  - [Learning From Prior Models](#learning-from-prior-models)
    - [Transfer Learning](#transfer-learning)
    - [Few-Shot Learning](#few-shot-learning)

<hr>

## Neural Architecture Search Problem
Neural Architecture Search (NAS) is a fundamental step in automating the machine learning process and has been successfully used to design the model architecture for image and language tasks.
  - 2018 | Progressive neural architecture search.  | Liu et al. | ECCV | [`PDF`](https://arxiv.org/abs/1712.00559)
  - 2018 | Efficient architecture search by network transformation. | Cai et al. | AAAI | [`PDF`](https://arxiv.org/abs/1707.04873)
  - 2018 | Learning transferable architectures for scalable image recognition. | Zoph et al. | IEEE CVPR | [`PDF`](https://arxiv.org/abs/1707.07012)
  - 2017 | Hierarchical representations for efficient architecture search. | Liu et al. | [`PDF`](https://arxiv.org/abs/1711.00436)
  - 2016 | Neural architecture search with reinforcement learning.  | Zoph and Le | [`PDF`](https://arxiv.org/abs/1611.01578)
  - 2009 | Learning deep architectures for AI. | Bengio et al. | [`PDF`](https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)

<div style="text-align: center">
<img src="Figures/NAS-1.png" width="700px" atl="Neural Architecture Search Methods"/>
</div>

+ ### Random Search
  - 2019 | Random Search and Reproducibility for Neural Architecture Search. | Li and Talwalkar | [`PDF`](https://arxiv.org/abs/1902.07638)
  - 2017 | Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks. | Hoffer et al.  | NIPS | [`PDF`](https://arxiv.org/abs/1705.08741)
+ ### Reinforcement Learning
  - 2019 | Neural architecture search with reinforcement learning. | Zoph and Le | [`PDF`](https://arxiv.org/abs/1611.01578)
  - 2019 | Designing neural network architectures using reinforcement learning. | Baker et al. | [`PDF`](https://arxiv.org/abs/1611.02167)
+ ### Evolutionary Methods
  - 2019 | Evolutionary Neural AutoML for Deep Learning. | Liang et al. | [`PDF`](https://arxiv.org/abs/1902.06827)
  - 2019 | Evolving deep neural networks. | Miikkulainen et al. | [`PDF`](https://arxiv.org/abs/1703.00548)
  - 2018 | a multi-objective genetic algorithm for neural architecture search. | Lu et al. | [`PDF`](https://arxiv.org/abs/1810.03522)
  - 2018 | Efficient multi-objective neural architecture search via lamarckian evolution. | Elsken et al. | [`PDF`](https://arxiv.org/abs/1804.09081)
  - 2018 | Regularized evolution for image classifier architecture search. | Real et al. | [`PDF`](https://arxiv.org/abs/1802.01548)
  - 2017 | Large-scale evolution of image classifiers | Real et al. | ICML | [`PDF`](https://arxiv.org/abs/1703.01041)
  - 2017 | Hierarchical representations for efficient architecture search. | Liu et al. | [`PDF`](https://arxiv.org/abs/1711.00436)
  - 2009 | A hypercube-based encoding for evolving large-scale neural networks. | Stanley et al. | Artificial Life | [`PDF`](http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf)
  - 2002 | Evolving neural networks through augmenting topologies. | Stanley and Miikkulainen | Evolutionary Computation | [`PDF`](https://dl.acm.org/citation.cfm?id=638554)

  <hr>

## AutoML Tools and Frameworks

  + ### Distributed Frameworks
  
  |               | Date | Language |    Training Framework   |                  Optimization Method                  | Meta-Learning | UI |                                          Open Source                                          |                               PDF                              |
|:-------------:|:----:|:--------:|:-----------------------:|:-----------------------------------------------------:|:-------------:|:--:|:---------------------------------------------------------------------------------------------:|:--------------------------------------------------------------:|
|     MLBase    | 2013 |   Scala  |          SparkMlib         |             Cost-based Multi-Armed Bandits            |       ×       |  × |                             × [`Website`](http://www.mlbase.org/)                             | [`PDF`](http://cidrdb.org/cidr2013/Papers/CIDR13_Paper118.pdf) |
|      ATM      | 2017 |  Python  |         Scikit-Learn       | Hybrid Bayesian, and Multi-armed bandits Optimization |       √       |  × |                         [`Github`](https://github.com/HDI-Project/ATM)                        |            [`PDF`](https://cyphe.rs/static/atm.pdf)            |
|     MLBox     | 2017 |  Python  |      Scikit-Learn Keras    | Distributed Random search, and Tree-Parzen estimators |       ×       |  × |                       [`Github`](https://github.com/AxeldeRomblay/MLBox)                      |                                ×                               |
|     Rafiki    | 2018 |  Python  |   Scikit-Learn TensorFlow  |    Distributed random search, Bayesian Optimization   |       ×       |  √ |                          [`Github`](https://github.com/nginyc/rafiki)                         |     [`PDF`](http://www.vldb.org/pvldb/vol12/p128-wang.pdf)     |
| TransmogrifAI | 2018 |   Scala  |           SparkML          |        Bayesian Optimization, and Random Search       |       ×       |  × | [`Github`](https://github.com/salesforce/TransmogrifAI)  [`Website`](https://transmogrif.ai/) |                                ×                               |
|    ATMSeer    | 2019 |  Python  | Scikit-Learn On Top Of ATM | Hybrid Bayesian, and Multi-armed bandits Optimization |       √       |  √ |                         [`Github`](https://github.com/HDI-Project/ATMSeer)                    |            [`PDF`](https://arxiv.org/abs/1902.05009)           |
|    D-SmartML  | 2019 |  Scala   | SparkMlib                  | Grid Search, Random Search, Hyperband                 |       √       |  x |                         [`Github`](https://github.com/DataSystemsGroupUT/Distributed-SmartML) |            x                                                   |
|   Databricks  | 2019 |  Python  | SparkMlib                  | Hyperopt                                              |       x       |  √ |                         × [`Website`](https://databricks.com/product/automl-on-databricks#resource-link) |            x                                                   |
<hr>


### Session-based Recommendation Datasets
  - 2019 | Third AutoML Challenge | [`URL`](https://competitions.codalab.org/competitions/19836)
  - 2018 | Second AutoML Challenge | [`URL`](https://competitions.codalab.org/competitions/17767)
  - 2017 | First AutoML Challenge | [`URL`](https://competitions.codalab.org/competitions/2321)

<hr>

## Contribute:  
To contribute a change to add more references to our repository, you can follow these steps:
1. Create a branch in git and make your changes.
2. Push branch to github and issue pull request (PR).
3. Discuss the pull request.
4. We are going to review the request, and merge it to the repository.

<hr>


## Citation:
 For more details, please refer to our benhcmarking Paper [`PDF`](#)
 ```
authors., Comprehensive Evaluation of Deep Learning Approaches for Session-based Recommendation in E-Commerce (2020).
```
