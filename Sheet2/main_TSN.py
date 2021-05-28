import torch
from TSN.dataset import RGBDataset
from TSN.model import TSNRGBModel
from datetime import datetime


no_segments=4
batch_size=64

dataset = RGBDataset(training=True, no_segments=no_segments)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

TSN_rgb_model = TSNRGBModel()
for data, labels in dataloader:
    data_ = data.view(batch_size*data.shape[1], data.shape[2],data.shape[3],data.shape[4])
    o = TSN_rgb_model(data_)
    print(data.shape)
    print(data_.shape)
    print(o.shape)
    break
    

    