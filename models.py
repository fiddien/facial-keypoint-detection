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
        ## it's suggested that you make this last layer output 136 values, 
        ## 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, 5)
        # input: (1, 224, 224)
        # conv1: 5x5 conv filter, 16 feature maps
        # output: (8, 220, 220)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, 
        # and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.pool = nn.MaxPool2d(2, 2)
        # pool: 2 kernel, 2 stride
        # output: (8, 110, 110)
        
        self.conv2 = nn.Conv2d(8, 16, 4)
        # conv2: 3x3 conv kernal, 16 feature maps
        # output: (16, 116, 116)
        
        # pool: 2 kernal, 2 stride
        # output: (16, 58, 58)
        
        self.conv3 = nn.Conv2d(16, 32, 3)
        # conv3: 3x3 conv kernel, 64 feature maps
        # output: (32, 56, 56)
        
        # pool: 2 kernal, 2 stride
        # output: (32, 28, 28)
        
        self.conv4 = nn.Conv2d(32, 64, 2)
        # conv3: 3x3 conv kernel, 128 feature maps
        # output: (64, 26, 26)
        
        # pool: 2 kernal, 2 stride
        # output: (64, 13, 13)
        
        self.drop = nn.Dropout(0.3)
        # output: (64, 13, 13)
       
        # self.flat = torch.Flatten()
        # output: (64*13*13)
        
        self.fc1 = nn.Linear(2304, 544)
        # output: (256)
        self.fc2 = nn.Linear(544, 272)
        # output: (128)
        self.fc3 = nn.Linear(272, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.pool(F.elu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.elu(self.fc1(x)))
        x = self.drop(F.elu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
