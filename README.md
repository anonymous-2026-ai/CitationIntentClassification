# Overview
This repository contains the implementation for our submitted manuscript. We are actively refining the codebase, improving documentation, and refactoring for better clarity during the peer-review process.

# Requirements
To run our system, you only need to install some very basic libraries such as: pytorch, transformers = 4.32.1. 

Besides, to run the clustering algorithm, you need to install and download 2 additional libraries for k-means and GMM from the following links:

1) k-means: https://github.com/subhadarship/kmeans_pytorch

2) GMM: https://github.com/ldeecke/gmm-torch

# Preparing Datasets

You can download the datasets from the following links:

1) Multicite: [https://github.com/allenai/multicite](https://github.com/allenai/multicite/tree/master/data/classification_gold_context)
   
2) ACL-ARC, SciCite datasets: https://github.com/allenai/scicite. 

However, after downloading the files, you have to convert them from .jsonl format to .json format to use them with our system. 

If you do not want to make any additional preprocessing steps, you can run our code directly with our processed datasets.

# Training 

Step 1: You can go to the folder for the specific dataset to run the training/testing code. 

Step 2: Look at the run.sh file and change the path to your dataset with '--data_dir' parameter. You can set up your own hyperparameters to start the training process. 

Step 3: To train, you run: bash run.sh.

Note that: In this version, the number of clusters $K$ is fixed in model.py, but you can easily change it. We will continue refactoring the code to enhance convenience. Besides, each training and testing run was conducted on a single GPU. 

# Testing 

Step 1: Select the optimal checkpoint based on your preferred performance metrics and transfer it from the train_folder to the test_folder for evaluation.

Step 2: In the test_folder, please name the test model as 'model_test'. 

Step 3: After that, you have to set up the value of the hyperparameters of the test model in 'run.sh' file to start the testing process. 

Step 4: To test, you run: bash run.sh.  
