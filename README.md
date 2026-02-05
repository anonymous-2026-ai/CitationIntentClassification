# Overview
This is the implementation for our work: **SynIntent**

# Requirements
To run our system, you only need to install some very basic libraries such as: pytorch, transformers = 4.32.1. 

Besides, to run the clustering algorithm, you need to install 2 additional libraries for k-means and GMM:

1) k-means: https://github.com/subhadarship/k-meanss_pytorch

2) GMM: https://github.com/ldeecke/gmm-torch

# Preparing Dataset 

You can download from the following links as follows:

1) Multicite: [https://github.com/allenai/multicite](https://github.com/allenai/multicite/tree/master/data/classification_gold_context)
   
2) ACL-ARC, SciCite dataset: https://github.com/allenai/scicite

# Training 

You can go to the folder for the specific dataset to run the training/testing code. Look at the run.sh file and change the path to your dataset with '--data_dir' parameter. You can set up your own hyperparameters to start the training process by running: bash run.sh.

# Testing 

You need to copy the checkpoint you want to test from train_folder to the test_folder. In the test_folder, please name the test model as 'model_test'. After that, you can start the testing process by running: bash run.sh. 
