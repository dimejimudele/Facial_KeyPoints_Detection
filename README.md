# FACIAL KEYPOINT DETECTION 
## Udacity computer vision nanodegree Project 1

This project is about defining and training a convolutional neural network to perform facial keypoint detection, 
and using computer vision techniques to transform images of faces. The implementation is done using PyTorch framework.


Facial keypoints (also called facial landmarks) are the numerically annotated small dots shown in the image below. 
In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. 
These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. 
These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. 
Here they are, numbered, and you can see that specific ranges of points match different portions of the face.



<div>
<img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/landmarks_numbered.jpg" width="200" height="300"/>
</div>

## Data
The set of image data used to train the CNN in this project have been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. 
These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

#### Training and Testing Data
This facial keypoints dataset consists of 5770 color images. 
All of these images are separated into either a training or a test set of data.

3462 of these images are training images, used to create the keypoint prediction model.
2308 are test images, which will be used to test the accuracy of the model.
The information about the images and keypoints in this dataset are summarized in CSV files, read in using pandas library. 


## Network achitecture

The convolutional neural network is defined as class `Net()` in the file named `model.py` with the archetecture shown below:
```
Net(
  (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (conv5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=12800, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=136, bias=True)
  (dropout): Dropout(p=0.3)
)
```

## Parameter settings and evaluation metrics

The loss function and optimizer are defined in PyTorch as below:

```
import torch.optim as optim

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr = 0.0001, weight_decay = 0)
```
As shown above, I used the mean sqaure error loss function and Adam optimizer. I trained the model for *5 epochs*

## Results
#### Before training
Random images selected
<div class="row">
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/pretrain_1.png">
  </div>
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/pretrain_2.png">
  </div>
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/pretrain_3.png">
  </div>
</div>

#### After training
Random images selected
<div class="row">
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/test_1.png">
  </div>
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/test_2.png">
  </div>
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/test_3.png">
  </div>
</div>

The best model obtained during the 5 epoch training was used for the predictions shown above. This model has been saved in the file: 
`.\saved_models\keypoints_project_CNN_model.pt` of the project directory.

## Real world application
Since real world images do not only contain faces, there is the need to extract faces from them before applying the CNN model obtained in this project.
For this, a pretrained Haar Cascade detector can be used. I outline the process for doing this below:

* Detect all the faces in an image using a face detector. For this project, I used 
[Haar Cascade detector](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) pretrained model.
* Pre-process those face images so that they are grayscale, and transformed to a Tensor of the input size fit for the CNN. 
Preprocessing tasks performed include rescaling, normalization, and conversion of images to Tensor.
* Use your trained model to detect facial keypoints on the image.

#### Sample result compatible with real world application

<div class="row">
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/obamas.jpg" width="400" height="300"/>
  </div>

  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/obamas_detected.png" width="400" height="300"/>
  </div>

  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/michelle_detected.png" width="300" height="300"/>
  </div>
  <div class="column">
    <img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/barack_detected.png" width="300" height="300"/>
  </div>
</div>
