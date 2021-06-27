import torch
from torchvision.io.video import read_video
from PIL import Image
import numpy as np

from os import listdir
from os.path import isfile, join
import torchvision.transforms.functional as tf



class TCNDataset(torch.utils.data.Dataset):
    def __init__(self,path='../data' , training = True):
        
        self.path = path
        # load classes mapping
        self.class2index = {} 
        self.index2class = {}
        classes_file = open(path+'/mapping.txt','r')
        for line in classes_file:
            line = line.rstrip('\n') 
            splitted = line.split(' ')
            self.class2index[splitted[1]] = splitted[0]
            self.index2class[splitted[0]] = splitted[1]
        classes_file.close()
        
        # load class names 
        video_list_location = '/train.bundle' if training else '/test.bundle'
        video_list_file = open(path+video_list_location)
        self.video_list = []
        for line in video_list_file:
            line = line.rstrip('\n')
            name = line.split('.txt')[0]
            self.video_list.append(name)
            
        video_list_file.close()
        
        self.size = len(self.video_list)
        self.training = training

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        
        video_name = self.video_list[index]
        features = torch.from_numpy(np.load(self.path+'/features/'+video_name+'.npy'))
        labels_file = open(self.path+'/groundTruth/'+video_name+'.txt')
        labels = torch.zeros((features.shape[1],))
        for idx, line in enumerate(labels_file):
             line = line.rstrip('\n')
             label = self.class2index[line]
             labels[idx] = int(label)
        labels_file.close()
        return features, labels

                