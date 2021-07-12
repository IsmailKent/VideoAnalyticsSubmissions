import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import Dataset
from model import HMM , GMM , SimpleMLP , RNN
from util import collate_fn_padd_training, collate_fn_padd_test , get_subaction_alignment , get_transitions_prior


    

    


def train_HMM_with_GMM(dataloader):
    num_subactions = num_states* num_actions
    video1, labels1, _ , transcripts1 =next(iter(dataloader))
    hmm = HMM()
    gmm = GMM(video1)
    # FOR EACH VIDEO:
    # GET ALIGNMENT
    # DIVIDE ACTIONS TO SUBACTION
    # CREATE #action*#states subactions uniformly
    # create HMM using states and grammar
    # apply HMM learning and GMM
    # from GMM get new alignment
    # repeat
    for video, labels, _ , transcripts in dataloader:
        alignment = labels
        #initialize probabilies between one action to move forward, but last subaction and first action in each actior according to grammar 
        A_prior = get_transitions_prior(class2index)
        hmm.A = A_prior
        # trianing 5 times like states in sheet
        for _ in range(5):
            subaction_alignments = get_subaction_alignment(alignment)
            #next:
                    # apply HMM learning and GMM
                    # from GMM get new alignment
            
        
        
    return hmm, gmm 
    
num_actions= 12
num_states = 16
batch_size = 1


training_dataset = Dataset(training=True)
class2index = training_dataset.class2index
training_dataloader = torch.utils.data.DataLoader(training_dataset,collate_fn=collate_fn_padd_training,  batch_size=batch_size, shuffle=False, drop_last=False)

test_dataset = Dataset(training=False)
test_dataloader = torch.utils.data.DataLoader(training_dataset,collate_fn=collate_fn_padd_test,  batch_size=batch_size, shuffle=False, drop_last=False)

for features,labels , masks, transcripts  in training_dataloader:
    print(features.shape, labels.shape, masks.shape, len(transcripts))
    break
    
for features,labels, masks, in test_dataloader:
    print(features.shape, labels.shape , masks.shape)
    break





