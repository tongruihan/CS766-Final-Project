# CS766-Final-Project
This repo contains the codes we developed and used for CS 766 course project: Face Mask Detection Model and Robustness Improvement by AUROC Optimization.
Our aim is to successfully detect whether people in sight wear face masks or not through the deep learning models.
We developed two different types of models: classifier + face detector and face mask detector.
Additionally, we present a new metrics of improving the Area Under Receiver Operating Characteristic Curve as an substitute for model training objective function.


## Classification Model

## YOLO

## AUROC Optimization
Under the folder of AUROC, we developed the codes for model training in PyTorch. 

* dataset.py - It provides a MaskDataset derived from the base Dataset class from PyTorch which loads our dataset.

* example.py - In this file, we generated the synthetic toy dataset as mentioned in the project website and train a two-layer neural network to improve AUROC value.

* train.py - This is the main file for training a pre-trained MobileNetv2 for the objective of AUROC optimization. In addition, we have tried ResNet as well, which could be loaded from ResNet.py.

* loss.py, loss_utils.py - These two files deal with the formulation of new loss function for AUROC optimization, and are compatible with cuda operations.

In order to perform MobileNetv2 training, you can simply run
```
python train.py
```
