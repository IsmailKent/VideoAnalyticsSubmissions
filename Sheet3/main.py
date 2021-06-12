from dataset import JHMDB_Dataset
import torch
from PIL import Image
from matplotlib import pyplot




batch_size = 1

dataset = JHMDB_Dataset()
dataloader= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

for images, (bbox, action) in dataloader:
    break