# Comprehensive Benchmarking of Session-based Recommendation using Deep-Learning Approaches

<div style="text-align: center">
<img src="Figures/sessionreco.png" width="900px" atl="Machine Learning Pipeline"/>
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

## List of Papers

+ ### Surveys and Benchmarks
  - 2018 | Progressive neural architecture search.  | Liu et al. | ECCV | [`PDF`](https://arxiv.org/abs/1712.00559)
  - 2018 | Efficient architecture search by network transformation. | Cai et al. | AAAI | [`PDF`](https://arxiv.org/abs/1707.04873)
  - 2018 | Learning transferable architectures for scalable image recognition. | Zoph et al. | IEEE CVPR | [`PDF`](https://arxiv.org/abs/1707.07012)
  - 2017 | Hierarchical representations for efficient architecture search. | Liu et al. | [`PDF`](https://arxiv.org/abs/1711.00436)
  - 2016 | Neural architecture search with reinforcement learning.  | Zoph and Le | [`PDF`](https://arxiv.org/abs/1611.01578)
  - 2009 | Learning deep architectures for AI. | Bengio et al. | [`PDF`](https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)
  
+ ### Baselines
  - 2019 | Random Search and Reproducibility for Neural Architecture Search. | Li and Talwalkar | [`PDF`](https://arxiv.org/abs/1902.07638)
  - 2017 | Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks. | Hoffer et al.  | NIPS | [`PDF`](https://arxiv.org/abs/1705.08741)
  
+ ### Deep Learning in Generalized Session-based Recommendation
  - 2017 | 3d convolutional networks for session-based recommendation with content features. | RecSys | [`PDF`](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p138-tuan.pdf)
  - 2019 | Simple convolutional generative network for next item recommendation. | WSDM | [`PDF`](https://arxiv.org/pdf/1808.05163.pdf)
  - 2019 | Session-based recommen-dation with graph neural networks. | AAAI | [`PDF`](https://arxiv.org/abs/1811.00855)
  - 2019 | A collaborativesession-based recommendation approach with parallel memory modules. | SIGIR | [`PDF`](https://dl.acm.org/doi/10.1145/3331184.3331210)
  
+ ### Deep Learning in Personalized Session-based Recommendation
  - 2019 | Neural architecture search with reinforcement learning. | Zoph and Le | [`PDF`](https://arxiv.org/abs/1611.01578)
  - 2019 | Designing neural network architectures using reinforcement learning. | Baker et al. | [`PDF`](https://arxiv.org/abs/1611.02167)
  <hr>

## Survey of Deep-Learning Approaches in session-based recommendation
  
 
<hr>


### E-Commerce Session-based Recommendation Datasets
  - 2015 | YOOCHOOSE - RecSys Challenge | [`URL`](http://2015.recsyschallenge.com/)
  - 2015 | Zalando Fashion Recommendation | [`NA`](https://zalando.com/)
  - 2016 | Diginetica - CIKM Cup | [`URL`](https://cikm2016.cs.iupui.edu/cikm-cup/)
  - 2016 | TMall (Taobao) - IJCAI16 Contest | [`URL`](https://tianchi.aliyun.com/dataset/dataDetail?dataId=53)
  - 2017 | Retail Rocket | [`URL`](https://www.kaggle.com/retailrocket/ecommerce-dataset)

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
