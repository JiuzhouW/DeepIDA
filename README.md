# DeepIDA
Deep Integrative Discriminant Analysis (Deep IDA) for Multi-view Data With Feature Ranking

This folder contains Python codes and a simple demostration of usage for the **Deep Integrative Discriminant Analysis** (Deep IDA) method presented in the article:

Jiuzhou Wang, Sandra E. Safo. [Deep IDA: A Deep Learning Method for Integrative Discriminant Analysis of Multi-View Data with Feature Ranking - An Application to COVID-19 severity](https://arxiv.org/abs/2111.09964), 2021.

To implement Deep IDA method for classification and Bi-Bootstrap for feature selection, three files are needed: helper_functions.py and utils.py contain the required helper functions; main_functions.py contain the major functions with more detailed descriptions:

1. DeepIDA_nonBootstrap: our proposed Deep IDA classfication without Bi-bootstrap feature selection. Input the features and training/tuning labels and it will return the classification accuracy/labels based on DeepIDA+NCC (Deep IDA with nearest centroid classification), SVM on stacked or individual data, and the variables selected by Deep IDA+TS (Deep IDA + Teacher Student Network).
2. DeepIDA_VS: our proposed Bi-bootstrap feature selection using Deep IDA. Input the features and training/tuning labels and it will return selected features (top 10% by default). Furthermore, it provides three files for each view: a) a csv file which containing the relative feature importance; b) a pdf of graph of all feature importance; c) a pdf of graph of top 10 percent of variables' feature importance.
3. DeepIDA_Bootstrap: our proposed Deep IDA with Bi-bootstrap feature selection. It combines the previous two functions and have the outputs mentioned in DeepIDA_VS and DeepIDA_nonBootstrap but based on selected features.

The file reproducibility demonstrates two exmaples. The MNIST and simulation examples required are [provided](https://drive.google.com/drive/folders/1e9Bt1jaSZIcgOWWTDq9wHOoaWmmGlFIb). All documented described are identical to the previous [github page](https://github.com/lasandrall/DeepIDA). Here for further simplity, we demonstrate a quick use of our method.


## check Python version and required package
```
from platform import python_version
print(python_version())
import torch
import numpy as np
import pandas as pd
```
The functions are built on 3.8.3. 

## download the codes and data and load them into the environment

Download main_functions, helper_functions.py, utils.py and 9 csv files.
```
Y_train = torch.tensor(pd.read_csv("YTrain.csv", header = None).values).reshape(1,-1)[0]
Y_valid = torch.tensor(pd.read_csv("YTune.csv", header = None).values).reshape(1,-1)[0]
Y_test = torch.tensor(pd.read_csv("YTest.csv", header = None).values).reshape(1,-1)[0]
data1_train = torch.tensor(pd.read_csv("X1Train.csv", header = None).values).float()
data2_train = torch.tensor(pd.read_csv("X2Train.csv", header = None).values).float()
data1_valid = torch.tensor(pd.read_csv("X1Tune.csv", header = None).values).float()
data2_valid = torch.tensor(pd.read_csv("X2Tune.csv", header = None).values).float()
data1_test = torch.tensor(pd.read_csv("X1Test.csv", header = None).values).float()
data2_test = torch.tensor(pd.read_csv("X1Test.csv", header = None).values).float()
data_train = [data1_train,data2_train]
data_valid = [data1_valid,data2_valid]
data_test = [data1_test,data2_test]
```

## set the hyperparameters
```
# structure: a list of arrays. Each array specifies the network structure (number of nodes in each layer) beyond the input layer 
# e.g. Construct "input-256-64-20" network structure for three views: [[256,64,20],[256,64,20],[256,64,20]]
structure = [[256,256,256,256,256,256,256,256,256,256,64,20],
             [256,256,256,256,256,256,256,256,256,256,64,20]]
# lr_rate: learning rate for each optimization step (Adam is used)
learning_rate = 0.01
# n_epoch: number of epochs to train the model
n_epoch = 30
# TS_num_features: an array. Each value represents number of features to select by Teacher Student Method on one view
TS_num_features = [50,50]
```

## run Deep IDA classification

The computational time for Deep IDA is usually 10s for feature size less than 1000. If Bootstrap version of Deep IDA is used for feature size greater than 2000, recommend to use parallel computing. For both versions, we provide classfication accuracy as the default metrics but one can compute other desired metrics as we output the class assigments as well.

### Deep IDA without feature selection

```
# Non-Bootstrap version of DeepIDA method

# Input:
# data_train: a list of tensors. Each tensor is the training data for one view 
# data_valid: a list of tensors. Each tensor is the validation data for one view
# data_test: a list of tensors. Each tensor is the test data for one view
# Y_train/valid/test: a tensor representing the group belongings for training/validation/test data


# Output:
# DeepIDA_train/test_acc: classification accuracy based on DeepIDA model trained by each view
# DeepIDA_train/test_labels: training/test data labels predicted by DeepIDA classification model trained by individual view
# DeepIDA_train/test_acc_com: DeepIDA classification accuracy based on combined last layers from all views
# DeepIDA_train/test_labels_com: 
#   training/test data labels predicted by DeepIDA classification model trained by combined last layers from all views
# SVM_train/test_acc: classification accuracy based on SVM trained by each view
# SVM_train/test_labels: training/test data labels predicted by SVM trained by individual view
# SVM_train/test_acc_com: classification accuracy based on SVM trained by stacked all views 
# SVM_train/test_labels_com: training/test data labels predicted by SVM trained by stacked all views
# TS_selected: features selected by Teacher Student Method

from main_functions import DeepIDA_nonBootstrap
result_nonBoot = DeepIDA_nonBootstrap(data_train, data_valid, data_test, Y_train, Y_valid, Y_test, structure, TS_num_features,  learning_rate, n_epoch)

# Based on all variables
# Training/Test classification rate for DeepIDA based on all variables
print("Training classification accuracy DeepIDA on all views: "+str(result_nonBoot[0]))
print("Test classification accuracy DeepIDA on all views: "+str(result_nonBoot[1]))
for i in range(len(data_train)):
    print("Training classification accuracy DeepIDA on view"+str(i+1)+":")
    print(result_nonBoot[4][i])
    print("Test classification accuracy DeepIDA on view"+str(i+1)+":")
    print(result_nonBoot[5][i])
```

The expected classfication accuracies for test data would be 0.68 (based on both views), 0.85 (based on view 1) and 0.50 (based on view 2) since view 2 data is pure noises. Different network initialization may lead to different outputs.

### Deep IDA with feature selection

First, some extra parameters need to be set:
```
# n_boot: number of bootstrap 
n_boot = 100
# variables_name_list: a list of arrays. Each array contains the variables's names for each view
variables_name_list = [pd.Index(['View1Var%d' % i for i in range(500)]),pd.Index(['View2Var%d' % i for i in range(500)])]
# n_epoch_boot: number of epochs to train the model
n_epoch_boot = 20
# n_epoch_nonboot: number of epochs to train the model
n_epoch_nonboot = 30
```

```
# Bootstrap version of DeepIDA method

# Input:
# data_train: a list of tensors. Each tensor is the training data for one view 
# data_valid: a list of tensors. Each tensor is the validation data for one view
# data_test: a list of tensors. Each tensor is the test data for one view
# Y_train/valid/test: a tensor representing the group belongings for training/validation/test data
# structure: a list of arrays. Each array specifies the network structure (number of nodes in each layer) beyond the input layer 
# e.g. Construct "input-256-64-20" network structure for three views: [[256,64,20],[256,64,20],[256,64,20]]
# TS_num_features: an array. Each value represents number of features to select by Teacher Student Method on one view
# lr_rate: learning rate for each optimization step (Adam is used)


# Output:
# DeepIDA_train/test_acc: classification accuracy based on DeepIDA model trained by each view
# DeepIDA_train/test_labels: training/test data labels predicted by DeepIDA classification model trained by individual view
# DeepIDA_train/test_acc_com: DeepIDA classification accuracy based on combined last layers from all views
# DeepIDA_train/test_labels_com: 
#   training/test data labels predicted by DeepIDA classification model trained by combined last layers from all views
# SVM_train/test_acc: classification accuracy based on SVM trained by each view
# SVM_train/test_labels: training/test data labels predicted by SVM trained by individual view
# SVM_train/test_acc_com: classification accuracy based on SVM trained by stacked all views 
# SVM_train/test_labels_com: training/test data labels predicted by SVM trained by stacked all views
# TS_selected: features selected by Teacher Student Method

from main_functions import DeepIDA_Bootstrap
result = DeepIDA_Bootstrap(data_train, data_valid, data_test, Y_train, Y_valid, Y_test,structure,n_boot, n_epoch_boot,variables_name_list, TS_num_features,  learning_rate, n_epoch_nonboot )
```

By default we select top 10% of the variables for each view. It provides three files for each view: a) a csv file which containing the relative feature importance; b) a pdf of graph of all feature importance; c) a pdf of graph of top 10 percent of variables' feature importance. Similar classification rates as non-bootstrap are expected to observe.











