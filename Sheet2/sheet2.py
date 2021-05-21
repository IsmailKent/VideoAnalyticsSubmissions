import torch
from TSN.dataset import RGBDataset

dataset = RGBDataset(training=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

for data, labels in dataloader:
    print(data.shape)
    break