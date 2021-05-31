import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.init import normal, constant
import numpy as np

class TSNRGBModel(torch.nn.Module):
    def __init__(self, num_class = 25, dropout = 0.8):
        super(TSNRGBModel, self).__init__()    

        self.num_class = num_class 
        self.dropout = dropout
        self.base_model = getattr(torchvision.models, 'resnet18')(True)
        self.base_model.last_layer_name = 'fc'

                
        self.feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        
        #adjust dropout like in paper
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(self.feature_dim, num_class))
            new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            new_fc = nn.Linear(self.feature_dim, num_class)
            
        # change last layer to get 25 classes for out training set 
        if (new_fc is not None):
            self.base_model.fc = new_fc
    
    def forward(self,x):
        x = self.base_model(x)
        return x

