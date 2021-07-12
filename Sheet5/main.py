import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import Dataset
from model import HMMGaussian , HMMwithLearning , SimpleMLP , RNN


# function for zero padding for dataloader because of variable video length
def collate_fn_padd_training(batch):
        batch_input , batch_target , transcripts = [list(t) for t in zip(*batch)] 
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), max(length_of_sequences),  np.shape(batch_input[0])[1], dtype=torch.float)
        
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        
        mask = torch.zeros(len(batch_input),  max(length_of_sequences), num_classes,dtype=torch.float)
 
        for i in range(len(batch_input)):
            batch_input_tensor[i, :np.shape(batch_input[i])[0],: ] = batch_input[i]
            
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = batch_target[i]
            
            mask[i, :np.shape(batch_target[i])[0], :] = torch.ones( batch_target[i].shape[0], num_classes)
            
        return batch_input_tensor, batch_target_tensor, mask , transcripts

def collate_fn_padd_test(batch):
        batch_input , batch_target , _ = [list(t) for t in zip(*batch)] 
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), max(length_of_sequences),  np.shape(batch_input[0])[1], dtype=torch.float)
        
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        
        mask = torch.zeros(len(batch_input),  max(length_of_sequences), num_classes,dtype=torch.float)
 
        for i in range(len(batch_input)):
            batch_input_tensor[i, :np.shape(batch_input[i])[0],: ] = batch_input[i]
            
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = batch_target[i]
            
            mask[i, :np.shape(batch_target[i])[0], :] = torch.ones( batch_target[i].shape[0], num_classes)
            
        return batch_input_tensor, batch_target_tensor, mask
   
    
num_classes= 12
batch_size = 4


training_dataset = Dataset(training=True)
training_dataloader = torch.utils.data.DataLoader(training_dataset,collate_fn=collate_fn_padd_training,  batch_size=batch_size, shuffle=True, drop_last=False)

test_dataset = Dataset(training=False)
test_dataloader = torch.utils.data.DataLoader(training_dataset,collate_fn=collate_fn_padd_test,  batch_size=batch_size, shuffle=True, drop_last=False)

for features,labels , masks, transcripts  in training_dataloader:
    print(features.shape, labels.shape, masks.shape, len(transcripts))
    break
    
for features,labels, masks, in test_dataloader:
    print(features.shape, labels.shape , masks.shape)
    break





