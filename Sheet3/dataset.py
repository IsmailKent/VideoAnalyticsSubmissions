import torch
from torchvision.io.video import read_video
from PIL import Image
import numpy as np

from os import listdir
from os.path import isfile, join
import torchvision.transforms.functional as tf

import glob
import json
from torchvision import transforms




# Returns images , (bbox, action number in int)
class JHMDB_Dataset(torch.utils.data.Dataset):
    def __init__(self,path='./data/release/train/'):
       
        folder_names = [folder.replace('\\','/') for folder in glob.glob(path+'*/') ] 
        
        self.image_paths = []
        self.labels = [] #list containing pairs (box, class) where box is [x , y , w, h]
        self.actions = dict() # dictionary of action_name : int 
        
        action_number =0
        for folder in folder_names:
            objects_file = open(folder+'objects.json')
            data = json.load(objects_file)
            for frame in data["frames"]:
                file_name = frame["file_name"]
                bbox = frame["bbox"]
                action_name = frame["action"]
                if action_name in self.actions:
                    action = self.actions[action_name]
                else:
                    action = action_number
                    action_number+=1
                    self.actions[action_name] = action
                
                self.image_paths.append(folder+file_name)
                self.labels.append((bbox,action))
            
                
            objects_file.close()
        self.size = len(self.image_paths)
        
        
        # default transforms for ResNet
        self.transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
        

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transforms(image)
        return image , self.labels[index]
        

                
                
