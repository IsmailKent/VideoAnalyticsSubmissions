import torch
from TSN.dataset import RGBDataset , OpticalFlowDataset
from TSN.model import TSNRGBModel , TSNFlowModel
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf




def train(model,dataloader,optimizer):
    model.train(True)
    i = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()


    for data, labels in dataloader:
        optimizer.zero_grad()
        data_ = data.view(batch_size*data.shape[1], data.shape[2],data.shape[3],data.shape[4]).cuda()
        labels_ = labels.view((labels.shape[0]*labels.shape[1])).cuda()
        predictions = model(data_)
        loss = criterion(predictions, labels_)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print("    Batch {}: loss = {}".format(i ,loss.item()))
        i += 1
    # return average loss
    return running_loss / i 


def accuracy(model, dataloader):
    num_correct = 0
    num_samples=0
    for data, labels in dataloader:
        data_ = data.view(batch_size*data.shape[1], data.shape[2],data.shape[3],data.shape[4]).cuda()
        labels_ = labels.view((labels.shape[0]*labels.shape[1])).cuda()
        predictions = model(data_)
        predictions_normalized = F.softmax(predictions,dim=1)
        predicted_classes = predictions_normalized.argmax(dim=1)
        correct = (predicted_classes == labels_)
        num_correct += correct.sum().float()
        num_samples+= labels_.shape[0]
        print('num of samples : ' , num_samples)
        print('num_correct : ', num_correct)
        
    return num_correct/num_samples




epochs = 50
no_segments=4
batch_size=32





print("========STARTING RGB TRAINING==========")
dataset_rgb_training = RGBDataset(training=True, no_segments=no_segments)
dataloader_rgb_training = torch.utils.data.DataLoader(dataset_rgb_training, batch_size=batch_size, shuffle=True, drop_last=True)

dataset_rgb_testing = RGBDataset(training=False, no_segments=no_segments)
dataloader_rgb_testing = torch.utils.data.DataLoader(dataset_rgb_testing, batch_size=batch_size, shuffle=True, drop_last=True)

TSN_rgb_model = TSNRGBModel().cuda()
optimizer_rgb = torch.optim.Adam(TSN_rgb_model.parameters(),lr=5e-4,betas=(0.9,0.95))

avg_loss_per_epoch = []
for epoch in range(epochs):
    print('starting epoch for RGB training: {}'.format(epoch))
    
    loss = train(TSN_rgb_model , dataloader_rgb_training,  optimizer_rgb)
    avg_loss_per_epoch.append(loss)
    print('average loss on epoch {}: {}'.format(epoch,loss))
    print('evaluating on validation set')
    accuracy = accuracy(TSN_rgb_model, dataloader_rgb_testing)
    print('current accuracy: {}'.format(accuracy))


print("========STARTING FLOW TRAINING==========")

dataset_flow_training = OpticalFlowDataset(training=True, no_segments=no_segments)
dataloader_flow_training = torch.utils.data.DataLoader(dataset_flow_training, batch_size=batch_size, shuffle=True, drop_last=True)

dataset_flow_testing = OpticalFlowDataset(training=False, no_segments=no_segments)
dataloader_flow_testing = torch.utils.data.DataLoader(dataset_flow_testing, batch_size=batch_size, shuffle=True, drop_last=True)


TSN_flow_model = TSNFlowModel().cuda()
optimizer_flow = torch.optim.Adam(TSN_flow_model.parameters(),lr=5e-4,betas=(0.9,0.95))

for epoch in range(epochs):
    print('starting epoch for optical flow training: {}'.format(epoch))
    
    loss = train(TSN_flow_model , dataset_flow_training,  optimizer_flow)
    avg_loss_per_epoch.append(loss)
    print('average loss on epoch {}: {}'.format(epoch,loss))
    print('evaluating on validation set')
    accuracy = accuracy(TSN_flow_model, dataloader_flow_testing)
    print('current accuracy: {}'.format(accuracy))
    

    
    
    

    