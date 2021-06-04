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
from os import listdir
from os.path import isfile, join
import torchvision.transforms.functional as tf



class RGBDataset(torch.utils.data.Dataset):
    def __init__(self,path='./data/mini_UCF' , training = True , no_segments = 4):
        
        self.no_segments = no_segments
        # load text file
        self.classes = [] # read from classes text
        classes_file = open(path+'/classes.txt','r')
        for line in classes_file:
            # remove end of line char
            line = line.rstrip('\n') 
            splitted = line.split(' ')
            self.classes.append(splitted[1])
        classes_file.close()
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
        
        #cut each video to K segments, then extract a random snippit
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
        
        label = self.labels[index]
        # return same labels n_segments time for all segments for training purposes 
        returned_labels = [label] * self.no_segments
        
        return snippets, torch.LongTensor(returned_labels)

                
                

    
    
    
class OpticalFlowDataset(torch.utils.data.Dataset):
    # same as RGB class
    def __init__(self,path='./data/mini-ucf101_flow_img_tvl1_gpu' , training = True , no_segments = 4):
        
        # load text file
        self.classes = [] # read from classes text
        classes_file = open('./data/mini_UCF/classes.txt','r')
        for line in classes_file:
            # remove end of line char
            line = line.rstrip('\n') 
            splitted = line.split(' ')
            self.classes.append(splitted[1])
        classes_file.close()
        frames = []
        labels = []
        

        
        ## keep list of video names and labels , only read them in get_item, to avoid very large memory needs.
        ## In get Item, do the computation of snipets  
        videos = []
        if (training):
            training_file = open('./data/mini_UCF/train.txt','r')
            for line in training_file:
                # remove end of line char
                line = line.rstrip('\n') 
                video_name = path+'/'+line
                videos.append(video_name)
                splitter = line.split('/')
                label = int(self.classes.index(splitter[0]))
                labels.append(label)

                
            training_file.close()
        
         
        else:
            

            validation_file =  open('./data/mini_UCF/validation.txt','r')
            for line in validation_file:
                # remove end of line char
                line = line.rstrip('\n') 
                video_name = path+'/'+line
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
        self.path = path

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        #cut each video to K segments, then extract equal number of random snippits
        #from each of them, and add these with the label 
        
        video_name = self.videos[index]
        #get list of all files in video folder
        
       
        flow_images = [f for f in listdir(video_name) if isfile(join(video_name, f))]
        
        # read first flow image to get dimensions
        first_image = tf.to_tensor(Image.open(video_name+'/'+flow_images[0]))
        
        length_video = len(flow_images) // 2
        length_of_segment = length_video // self.no_segments
        snippets = torch.zeros((self.no_segments,5*2, first_image.shape[1], first_image.shape[2]))
        for i in range(self.no_segments):

            start = i*length_of_segment
            for flow_idx in range(5):
                idx = start + flow_idx
                # add path here
                x_flow_image = Image.open(video_name+'/'+flow_images[idx])
                x_flow_image = tf.to_tensor(x_flow_image)
                
                y_flow_image = Image.open(video_name+'/'+flow_images[idx+length_video])
                y_flow_image = tf.to_tensor(y_flow_image)

                snippets[i][2*flow_idx] = x_flow_image
                snippets[i][2*flow_idx+1] = y_flow_image

        
        label = self.labels[index]
        # return same labels n_segments time for all segments for training purposes 
        returned_labels = [label] * self.no_segments
        
        return snippets, torch.LongTensor(returned_labels)
    
    
    
# A dataset class that returns both rgb and flow snippets together, to test fusion
class FusingValidationDataset(torch.utils.data.Dataset):
    # same as RGB class
    def __init__(self,path_rgb= './data/mini_UCF/classes.txt' , path_flow='./data/mini-ucf101_flow_img_tvl1_gpu' , no_segments = 4):
        
        # load text file
        self.classes = [] # read from classes text
        classes_file = open('./data/mini_UCF/classes.txt','r')
        for line in classes_file:
            # remove end of line char
            line = line.rstrip('\n') 
            splitted = line.split(' ')
            self.classes.append(splitted[1])
        classes_file.close()
        frames = []
        labels = []
        

        
        ## keep list of video names and labels , only read them in get_item, to avoid very large memory needs.
        ## In get Item, do the computation of snipets  
        videos_rgb = []
        videos_flow = []

        validation_file =  open('./data/mini_UCF/validation.txt','r')
        for line in validation_file:
            # remove end of line char
            line = line.rstrip('\n') 
            video_name_rgb = path_rgb+'/'+line
            video_name_flow = path_flow+'/'+line
            videos_rgb.append(video_name_rgb)
            videos_flow.append(video_name_flow)
            splitter = line.split('/')
            label = int(self.classes.index(splitter[0]))
    
            labels.append(label)

                
                
        validation_file.close()
        
        self.videos_rgb = videos_rgb
        self.videos_flow = videos_flow
        self.labels = labels
        self.size = len(self.videos)
        self.no_segments = no_segments
        self.path_rgb = path_rgb
        self.path_flow = path_flow

    def __len__(self):
        return self.size


    def __getitem__(self, index):

        

        # ============= EXTRACT RGB SNIPPETS ================

        video_name = self.videos_rgb[index]

        video_frames , _ , _ = read_video(video_name)
        length_video = video_frames.shape[0]
        length_of_segment = length_video // self.no_segments
        snippets_rgb = torch.zeros((self.no_segments,video_frames.shape[3], video_frames.shape[1], video_frames.shape[2]))
        for i in range(self.no_segments):

            start = i*length_of_segment
            finish= min( start + length_of_segment , length_video)
            # for testing index is middle of segment
            idx = int(start + length_of_segment//2)
            snippet = video_frames[idx].permute((2, 0, 1))
            snippets_rgb[i] = snippet
        



        # ============= EXTRACT FLOW SNIPPETS ================
        
        video_name_flow = self.videos_flow[index]
        #get list of all files in video folder
        
       
        flow_images = [f for f in listdir(video_name_flow) if isfile(join(video_name_flow, f))]
        
        # read first flow image to get dimensions
        first_image_flow = tf.to_tensor(Image.open(video_name_flow+'/'+flow_images[0]))
        
        length_video = len(flow_images) // 2
        length_of_segment = length_video // self.no_segments
        snippets_flow = torch.zeros((self.no_segments,5*2, first_image_flow.shape[1], first_image_flow.shape[2]))
        for i in range(self.no_segments):

            start = i*length_of_segment
            for flow_idx in range(5):
                idx = start + flow_idx
                # add path here
                x_flow_image = Image.open(video_name_flow+'/'+flow_images[idx])
                x_flow_image = tf.to_tensor(x_flow_image)
                
                y_flow_image = Image.open(video_name_flow+'/'+flow_images[idx+length_video])
                y_flow_image = tf.to_tensor(y_flow_image)

                snippets_flow[i][2*flow_idx] = x_flow_image
                snippets_flow[i][2*flow_idx+1] = y_flow_image

        
        label = self.labels[index]
        # return same labels n_segments time for all segments for training purposes 
        returned_labels = [label] * self.no_segments
        
        return (snippets_rgb, snippets_flow), torch.LongTensor(returned_labels)