# FACIAL KEYPOINT DETECTION 
## Udacity computer vision nanodegree Project 1

This project is about defining and training a convolutional neural network to perform facial keypoint detection, 
and using computer vision techniques to transform images of faces. 


Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. 
In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. 
These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. 
These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. 
Here they are, numbered, and you can see that specific ranges of points match different portions of the face.


<div>
<img src="https://github.com/dimejimudele/Udacity_Facial_KeyPoints_Detection/blob/master/images/landmarks_numbered.jpg" width="300" height="500"/>
</div>

## Data
The set of image data used in this project have been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. 
These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

### Training and Testing Data
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
