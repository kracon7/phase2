import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CorridorNet(nn.Module):
    def __init__(self):
        super(CorridorNet, self).__init__()

        self.resnet = models.resnet18(pretrained=True)

        self.predictor = nn.Sequential(
                nn.Linear(1000, 256),
                nn.ReLU(),
                nn.Linear(256, 4)
            )
        
    def forward(self, x):
        ft = self.resnet(x)
        out = self.predictor(ft)
        return out