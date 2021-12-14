import torch
from torch import nn

from torchvision import models

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=False, progress=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
