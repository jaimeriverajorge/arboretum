# Preparing the fine-tuned ResNet50 model for training
# on the facial keypoints dataset

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceKeypointResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(FaceKeypointResNet50, self).__init__()
        if pretrained == True:
            self.model = torch.hub.load(
                'pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        else:
            self.model = torch.hub.load(
                'pytorch/vision:v0.9.0', 'resnet50', pretrained=False)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')

        # change the final layer
        self.l0 = nn.Linear(2048, 136)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)

        return l0
