# CS766-Final-Project
This repo contains the codes we developed and used for CS 766 course project: Face Mask Detection Model and Robustness Improvement by AUROC Optimization.
Our aim is to successfully detect whether people in sight wear face masks or not through the deep learning models.
We developed two different types of models: classifier + face detector and face mask detector.
Additionally, we present a new metrics of improving the Area Under Receiver Operating Characteristic Curve as an substitute for model training objective function.


## Classification Model
For this method, we adopt the pipeline where the pre-trained face detector (as provided in face_detector/) extracts the recognizable faces in the image/frame, 
and the classifier we trained will be used to identify whether the detected face wears a face mask or not. We used MobileNet v2 as the base model for classifier, 
and for the purpose of efficient training, we loaded the pre-trained model using the updated parameters for the feature selection layers. 
That is, our training was conducted on the classifier blocks within MobileNetv2, specifically the fully connected layers with the final output class numbers as 2.

* clsfier_train.py - We provided the training codes for MobileNetv2 in TensorFlow and generating the training result plots.

* image_detector.py - We can detect the face masks in a given image via this file.

* video_detector.py - We can detect the face masks in a given video or through camera livestream via this file.

```
python clsfier_train.py --dataset xxx
python image_detector.py --image xxx.png
python video_detector.py --video xxx.mp4
python video_detector.py # If you need to perform detection on a camera livestream.
```

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
In order to evaluate the performance on synthetic toy dataset, you can simply run
```
python example.py
```
