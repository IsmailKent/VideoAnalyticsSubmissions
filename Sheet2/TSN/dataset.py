import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.io.video import read_video , read_video_timestamps
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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
        
        #cut each video to K segments, then extract equal number of random snippits
        #from each of them, and add these with the label 
        
        if (training):
            training_file = open(path+'/train.txt','r')
            for line in training_file:
                # remove end of line char
                line = line.rstrip('\n') 
                print(path+'/'+line+'.avi')
                video = read_video(path+'/'+line+'.avi')[0]
                print(video)
                length_video = video.shape[0]
                length_of_segment = length_video // no_segments
                snippits = [] 
                for i in range(0,length_video, length_of_segment):
                    index = np.random.randint(i , min(length_video, i+ length_of_segment))
                    snippit = video[index]
                    snippits.append(snippit)
                frames += snippits
                
                
                splitter = line.split('/')
                label = int(self.classes.index(splitter[0]))
    
                labels += [label] * no_segments
                
            training_file.close()
        

        
       # cut each video to no_segments segments, then extract a snippit
       # from each of them, and add these with the label
         
        else:
            

            validation_file =  open(path+'/validation.txt','r')
            for line in validation_file:
                # remove end of line char
                line = line.rstrip('\n') 
                video = read_video(path+'/'+line+'.avi')[0]
                length_video = video.shape[0]
                length_of_segment = length_video // no_segments
                snippits = [] 
                for i in range(0,length_video, length_of_segment):
                    # calculate middle of segment, edge case last segment shorter than others 
                    index = i + ( min(i+length_of_segment , length_video) - i) // 2
                    snippit = video[index]
                    snippits.append(snippit)
                frames += snippits
                splitter = line.split('/')
                label = int(self.classes.index(splitter[0]))
                labels += [label] * no_segments
                
                
            validation_file.close()
        
        self.frames = frames
        self.labels = labels
        self.size = len(self.frames)
        print(len(self.frames))

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        return self.frames[index], self.labels[index]
    
    
    
class OpticalFlowDataset(torch.utils.data.Dataset):
    def __init__(self,type):
        ## TODO
        pass

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        return self.training_images[index], self.training_labels[index]