import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.io.video import read_video , read_video_timestamps
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datetime import datetime
from sys import getsizeof
import gc
from torchvision import transforms


class RGBDataset(torch.utils.data.Dataset):
    def __init__(self,path='./data/mini_UCF' , training = True , no_segments = 4):
        
        # load text file
        self.classes = [] # read from classes text
        classes_file = open(path+'/classes.txt','r')
        for line in classes_file:
            # remove end of line char
            line = line.rstrip('\n') 
            splitted = line.split(' ')
            self.classes.append(splitted[1])
        classes_file.close()
        print(self.classes)
        frames = []
        labels = []
        

        
        ## keep list of video names and labels , only read them in get_item, to avoid very large memory needs.
        ## In get Item, do the computation of snipets  
        videos = []
        if (training):
            training_file = open(path+'/train.txt','r')
            for line in training_file:
                # remove end of line char
                line = line.rstrip('\n') 
                video_name = path+'/'+line+'.avi'
                videos.append(video_name)
                splitter = line.split('/')
                label = int(self.classes.index(splitter[0]))
                labels.append(label)

                
            training_file.close()
        
         
        else:
            

            validation_file =  open(path+'/validation.txt','r')
            for line in validation_file:
                # remove end of line char
                line = line.rstrip('\n') 
                video_name = path+'/'+line+'.avi'
                videos.append(video_name)
                splitter = line.split('/')
                label = int(self.classes.index(splitter[0]))
    
                labels.append(label)

                
                
            validation_file.close()
        
        self.videos = videos
        self.labels = labels
        self.size = len(self.videos)
        self.no_segments = no_segments
        self.training = training

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        
        #cut each video to K segments, then extract equal number of random snippits
        #from each of them, and add these with the label 
        
        video_name = self.videos[index]

        video_frames , _ , _ = read_video(video_name)
        length_video = video_frames.shape[0]
        length_of_segment = length_video // self.no_segments
        snippets = torch.zeros((self.no_segments,video_frames.shape[3], video_frames.shape[1], video_frames.shape[2]))
        for i in range(self.no_segments):
            if (self.training):        
                start = i*length_of_segment
                finish= min( start + length_of_segment , length_video)
                # for training index is random
                idx = np.random.randint(start, finish)
                snippet = video_frames[idx].permute((2, 0, 1))
                snippets[i] = snippet
            else:
                start = i*length_of_segment
                finish= min( start + length_of_segment , length_video)
                # for testing index is middle of segment
                idx = int(start + length_of_segment//2)
                snippet = video_frames[idx].permute((2, 0, 1))
                snippets[i] = snippet
        
        # make a one hot encoding for class
        labels = torch.zeros((len(self.classes),))
        labels[self.labels[index]] = 1 
        return snippets, self.labels[index]

                
                

    
    
    
class OpticalFlowDataset(torch.utils.data.Dataset):
    def __init__(self,type):
        ## TODO
        pass

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        return self.training_images[index], self.training_labels[index]