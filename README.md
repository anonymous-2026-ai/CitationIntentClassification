# Overview
This is the implementation for our work: SynIntent

# Preparing Dataset 

You can download the Multicite dataset at: https://github.com/allenai/multicite
and ACL-ARC, SciCite dataset at: https://github.com/allenai/scicite . 

# Training 

You can go to the folder for the specific dataset to run the training/testing code. Look at the run.sh file and change the path to your dataset with '--data_dir' parameter. You can set up your own hyperparameters to start the training process. 

# Testing 

You need to copy the checkpoint you want to test from train_folder to the test_folder. In the test_folder, please name the test model as 'model_test'. After that, you can start the testing process. 
