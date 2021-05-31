import torch
from TSN.dataset import RGBDataset , OpticalFlowDataset
from TSN.model import TSNRGBModel
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf




def train_rgb(model,dataloader,optimizer):
    model.train(True)
    i = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()


    for data, labels in dataloader:
        optimizer.zero_grad()
        data_ = data.view(batch_size*data.shape[1], data.shape[2],data.shape[3],data.shape[4])
        labels_ = labels.view((labels.shape[0]*labels.shape[1]))
        predictions = model(data_)
        print(predictions.shape)
        print(labels_.shape)
        loss = criterion(predictions, labels_)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print("    Batch {}: loss = {}".format(i ,loss.item()))
        i += 1
    # return average loss
    return running_loss / i 


def accuracy_rgb(model, dataloader):

    for data, labels in dataloader:
        num_correct = 0
        num_samples=0
        data_ = data.view(batch_size*data.shape[1], data.shape[2],data.shape[3],data.shape[4])
        labels_ = labels.view((labels.shape[0]*labels.shape[1]))
        predictions = model(data_)
        predictions_normalized = F.softmax(predictions,dim=1)
        predicted_classes = predictions_normalized.argmax(dim=1)
        correct = (predicted_classes == labels_)
        num_correct += correct.sum().float()
        num_samples+= labels_.shape[0]
        print('num of samples : ' , num_samples)
        print('num_correct : ', num_correct)
        
    return num_correct/num_samples

    


epochs = 20
no_segments=4
batch_size=32


dataset_flow_training = OpticalFlowDataset(training=True, no_segments=no_segments)
dataloader_flow_training = torch.utils.data.DataLoader(dataset_flow_training, batch_size=batch_size, shuffle=True, drop_last=True)

for data , labels in dataloader_flow_training:
    print(data.shape)
    print(labels.shape)


dataset_rgb_training = RGBDataset(training=True, no_segments=no_segments)
dataloader_rgb_training = torch.utils.data.DataLoader(dataset_rgb_training, batch_size=batch_size, shuffle=True, drop_last=True)

dataset_rgb_testing = RGBDataset(training=False, no_segments=no_segments)
dataloader_rgb_testing = torch.utils.data.DataLoader(dataset_rgb_testing, batch_size=batch_size, shuffle=True, drop_last=True)

TSN_rgb_model = TSNRGBModel()
optimizer = torch.optim.Adam(TSN_rgb_model.parameters(),lr=5e-4,betas=(0.9,0.95))

avg_loss_per_epoch = []
for epoch in range(epochs):
    """
    print('starting epoch: {}'.format(epoch))
    
    loss = train_rgb(TSN_rgb_model , dataloader_rgb_training,  optimizer)
    avg_loss_per_epoch.append(loss)
    print('average loss on epoch {}: {}'.format(epoch,loss))
    """
    print('evaluating on validation set')
    accuracy = accuracy_rgb(TSN_rgb_model, dataloader_rgb_testing)
    print('current accuracy: {}'.format(accuracy))

    
    
    
    

    