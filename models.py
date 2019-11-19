## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        

        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        """
        Input image size = 224 x 224
        Convulational layer output image width = height = (w-F)/S + 1
        
        W: input image width
        F: Filter size - width or height
        S: Stribe of convolution operation 
        
        Example of 2d convoluion layer: self.conv1 = nn.Conv2d(1, 32, 5)  # (w-F)/S + 1 = (224 - 5) + 1 = 220
        
        K: out_channels - depth of filters in convolutional layer 
        P: the padding
        The total number of weights in a convolutional layer = F*F*K
        
        
        """
        # Convolution layers
        
        # 1: Output = (W-F)/S + 1 = (224-5) + 1 = 220
        # After pool layer: (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5) 
        
        # 2: Output = (W-F)/S + 1 = (110-3) + 1 = 108
        # After pool layer: (64, 54, 54)        
        self.conv2 = nn.Conv2d(32, 64, 3)

        # 3: Output = (W-F)/S + 1 = (54-3) + 1 = 52
        # After pool layer: (128, 26, 26) 
        self.conv3 = nn.Conv2d(64, 128, 3) 
        
        # 4: Output = (W-F)/S + 1 = (26-3) + 1 = 24
        # After pool layer: (256, 12, 12)         
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # 5: Output = (W-F)/S + 1 = (12-3) + 1 = 10
        # After pool layer: (512, 5, 5)  
        self.conv5 = nn.Conv2d(256, 512, 3) 
        
        # Max pooling
        # Pool with kernel-size = 2 and stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(512*5*5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)
        
        # Dropout
        self.dropout = nn.Dropout(p = 0.3)
        

        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        #Fatten and run thorugh FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) 

        return x
