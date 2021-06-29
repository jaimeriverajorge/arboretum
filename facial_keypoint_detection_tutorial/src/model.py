# script to build the neural network model
# 3 convolutional layers and one fully connected layer

import torch.nn as nn
import torch.nn.functional as F

# nn.Module is the base class for all neural network
# models, each model made should subclass it


class FaceKeypointModel(nn.Module):

    def __init__(self):
        super(FaceKeypointModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        self.fc1 = nn.Linear(128, 30)
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        # applying ReLu activation and MaxPool after
        # every convolutional layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # underscore here used as throwaway variables,
        # we only want the first dimension of x.shape
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        # apply dropout once before fc layer
        x = self.dropout(x)
        # no activation function in fc layer, we directly
        # need the regressed coordinates of keypoints
        out = self.fc1(x)

        return out
