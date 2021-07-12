import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self,path='./Data' , training = True):
        
        self.path = path
        # load classes mapping
        self.class2index = {} 
        self.index2class = {}
        index=0
        classes_file = open(path+'/classes.txt','r')
        for line in classes_file:
            line = line.rstrip('\n')
            self.class2index[line] = index
            self.index2class[index] = line
            index+=1
        classes_file.close()
        
        # load class names 
        video_list_location = '/train_list.txt' if training else '/test_list.txt'
        video_list_file = open(path+video_list_location)
        self.video_list = []
        for line in video_list_file:
            line = line.rstrip('\n')
            self.video_list.append(line)
            
        video_list_file.close()
        
        self.size = len(self.video_list)
        self.training = training

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        
        video_name = self.video_list[index]
        features = torch.from_numpy(np.load(self.path+'/features/'+video_name+'.npy'))
        if self.training:
            labels_file = open(self.path+'/uniform_labels/'+video_name+'.txt')
            labels = torch.zeros((features.shape[0],))
            for idx, line in enumerate(labels_file):
                line = line.rstrip('\n')
                labels[idx] = int(line)
                
            transcript_file =  open(self.path+'/transcripts/'+video_name+'.txt')
            transcript = []
            for idx, line in enumerate(transcript_file):
                line = line.rstrip('\n')
                label = self.class2index[line]
                transcript.append( int(label))
            
            


        else:
            labels_file = open(self.path+'/test/'+video_name+'.txt')
            labels = torch.zeros((features.shape[1],))
            for idx, line in enumerate(labels_file):
                line = line.rstrip('\n')
                label = self.class2index[line]
                labels[idx] = int(label)
                
        labels_file.close()
        if self.training:
            return features, labels , torch.Tensor(transcript)
        else:
            return features, labels            
        

                