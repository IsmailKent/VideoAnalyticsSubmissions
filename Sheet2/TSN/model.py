import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TSNRGBModel(torch.nn.Module):
    def __init__(self):
        super(TSNRGBModel, self).__init__()    
        self.base_model = torchvision.models.resnet18(pretrained=True, progress=True)
    
    def forward(self,x):
        x = self.base_model(x)
        return x

