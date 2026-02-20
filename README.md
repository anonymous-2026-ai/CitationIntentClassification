# Overview
This repository contains the implementation for our submitted manuscript. We are actively refining the codebase, improving documentation, and refactoring for better clarity during the peer-review process.

# Requirements
To set up the reproducible research environment, you need to follow the following steps: 

1) Install Conda and Python 3.11:

   conda create -n syn_intent python=3.11

   conda activate syn_intent

2) Install Python packages:

   cd CitationIntentClassification/
   
   pip install -r requirements.txt

4) NLTK resource setup:
   
   python -c "import nltk; nltk.download('all')"

Besides, to run the clustering algorithm, we downloaded 2 additional libraries for k-means and GMM from the following links:

1) k-means: https://github.com/subhadarship/kmeans_pytorch

2) GMM: https://github.com/ldeecke/gmm-torch

and modified them for our implementation. You should use them directly in our folders for result reproduction.

# Preparing Datasets

You can download the datasets from the following links:

1) Multicite: [https://github.com/allenai/multicite](https://github.com/allenai/multicite/tree/master/data/classification_gold_context)
   
2) ACL-ARC, SciCite datasets: https://github.com/allenai/scicite. 

However, after downloading the files, you have to convert them from .jsonl format to .json format to use them with our system. 

If you do not want to make any additional preprocessing steps, you can run our code directly with our processed datasets.

Note that, in the 'datasets' folder, the folder 'data_multicite' refers to the full MultiCite dataset while the folder 'data_multicite_only_multi_intent' indicates the subset of MultiCite dataset which only contains the samples with more than one intent label in the gold label.

# Training 

First, we set up a single-GPU environment for each run by executing: export CUDA_VISIBLE_DEVICES=0 

to ensure the model only utilizes one specific device. You should follow the following steps to start the training process. 

Step 1: You can go to the folder for the specific dataset to run the training/testing code. 

Step 2: Look at the run.sh file and change the path to your dataset with '--data_dir' parameter. You can set up your own hyperparameters to start the training process. 

Step 3: To train, you run: bash run.sh.

Note that: In this version, the number of clusters $K$ is fixed in model.py, but you can easily change it. We will continue refactoring the code to enhance convenience. 

# Testing 
First, we set up a single-GPU environment for each run by executing: export CUDA_VISIBLE_DEVICES=0 

to ensure the model only utilizes one specific device.  You should follow the following steps to start the testing process. 

Step 1: Select the optimal checkpoint based on your preferred performance metrics and transfer it from the train_folder to the test_folder for evaluation. You also have to change the path to your dataset with '--data_dir' parameter in 'run.sh' file.  

Step 2: In the test_folder, please name the test model as 'model_test'. 

Step 3: After that, you have to set up the value of the hyperparameters of the test model in 'run.sh' file and exact the number of clusters: K of the test model in 'model.py' file to start the testing process. 

Step 4: To test, you run: bash run.sh.  
