import torch
from TSN.dataset import RGBDataset
from datetime import datetime

dataset = RGBDataset(training=True, no_segments=4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

print(datetime.now())
for data, labels in dataloader:
    # outputs shape ((batch_size, no_segments, W, H , 3 ))
    print(data.shape)
    
print(datetime.now())

    