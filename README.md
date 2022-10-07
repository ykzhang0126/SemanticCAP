# SemanticCAP
## Introduction
This is the source code for paper "SemanticCAP: Chromatin Accessibility Prediction Enhanced by Features Learning from a Language Model", Yikang Zhang, Xiaomin Chu, Yelu Jiang, Hongjie Wu and Lijun Quan.
Source data and model files are available at [BaiduNetDisk](https://pan.baidu.com/s/1P_Mfu3xE5_hrUWYk2vKCg) and its access code is "fs8a".

## Install
The code is mainly written in Python (3.7) using tensorflow(2.5.0) and pytorch(1.7.0). One can install the required modules by following instructions on website [https://tensorflow.google.cn/install/pip/](https://tensorflow.google.cn/install/pip) and [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

The Anaconda platform is highly recommended.

## Structure
Pretrain processes of data are included in folder "process_raw_data"; DNA language models are included in folder "language_model"; chromatin accessibility models are included in folder "access_model"; Some comparative models are available in folder "etc", such as lstm, conv or other models.
Most Configurations can be modified in "config.py".

## Usage
First, wrap the dataset in a folder called "data" at the root of the entire project. Then, execute the corresponding file as required. GPU will be used if exists.

### Language model
Modify the language-model-related options in file "config.py" and enter the command
```
python ./access_model/train_access.py
```

### How to train
Modify the dataset, parameter, train-related options in file "config.py" and enter the command
```
python ./access_model/train_access.py
```
### How to evaluate
Modify the dataset, parameter, evaluation-related options in file "config.py" and enter the command
```
python ./access_model/eval_access.py
```
