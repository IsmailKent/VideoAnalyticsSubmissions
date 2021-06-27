import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import TCNDataset
from model import TCN , MultiStageTCN , ParallelTCNs


def train(model,dataloader,optimizer):
    i=0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0 
    for features, labels, masks in dataloader:
        out = model(features,masks)
        optimizer.zero_grad()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
                print("    Batch {}: loss = {}".format(i ,loss.item()))
        i += 1
        running_loss = loss.item()
    return running_loss / len(dataloader)

    
def train_parallel(model, dataloader,optimizer):
    model.train()
    i=0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0 
    for features, labels, masks in dataloader:
        out1, out2, out3 ,out_average = model(features,masks)
        optimizer.zero_grad()
        loss1 = criterion(out1, labels)
        loss2 = criterion(out2, labels)
        loss3 = criterion(out3, labels)
        loss_average = criterion(out_average, labels)
        loss = loss1 + loss2 + loss3 + loss_average
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
                print("    Batch {}: combined loss = {}".format(i ,loss.item()))
        i += 1
        running_loss = loss.item()
    return running_loss / len(dataloader)



# function for zero padding for dataloader because of variable video length
# inspired by the code from the paper
def collate_fn_padd(batch):
        batch_input , batch_target = [list(t) for t in zip(*batch)] 
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        
        mask = torch.zeros(len(batch_input), num_classes, max(length_of_sequences), dtype=torch.float)
        
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = batch_input[i]
            
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = batch_target[i]
            
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(num_classes, batch_target[i].shape[0])
            
        return batch_input_tensor, batch_target_tensor, mask
            
            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 50
num_classes = 48


# DATA LOADERS 

training_dataset = TCNDataset(training=True)
training_dataloader = torch.utils.data.DataLoader(training_dataset,collate_fn=collate_fn_padd,  batch_size=batch_size, shuffle=True, drop_last=False)

test_dataset = TCNDataset(training=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset,collate_fn=collate_fn_padd,  batch_size=batch_size, shuffle=False, drop_last=False)


single_TCN = TCN()
single_TCN_optimizer = torch.optim.Adam(single_TCN.parameters(),lr=1e-4)

multi_stage_TCN = MultiStageTCN()
multi_stage_TCN_optimizer = torch.optim.Adam(multi_stage_TCN.parameters(),lr=1e-4)

parallel_TCNs = ParallelTCNs()
parallel_TCNs_optimizer = torch.optim.Adam(parallel_TCNs.parameters(),lr=1e-4)


# call training functions from above inside this loop
for epoch in range(epochs):
    print("RUNNING EPOCH: {}".format(epoch+1))
    #train(single_TCN,training_dataloader , single_TCN_optimizer )
    #train(multi_stage_TCN,training_dataloader , multi_stage_TCN_optimizer )
    
    train_parallel(parallel_TCNs, training_dataloader, parallel_TCNs_optimizer)
