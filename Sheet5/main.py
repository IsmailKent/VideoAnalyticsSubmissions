import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import Dataset
from model import HMM , GMM , SimpleMLP , RNN
from utils import collate_fn_padd_training, collate_fn_padd_test , get_subaction_alignment , get_transitions_prior


    

    


def train_HMM_with_GMM(dataloader):
    num_subactions = num_states* num_actions
    video1, labels1, _ , transcripts1 =next(iter(dataloader))
    hmm = HMM()
    gmm = GMM(video1[0])
    # FOR EACH VIDEO:
    # GET ALIGNMENT
    # DIVIDE ACTIONS TO SUBACTION
    # CREATE #action*#states subactions uniformly
    # create HMM using states and grammar
    # apply HMM learning and GMM
    # from GMM get new alignment
    # repeat
    for video, labels, _ , transcripts in dataloader:
        video=video[0]
        alignment = get_subaction_alignment(labels.T)
        #initialize probabilies between one action to move forward, but last subaction and first action in each actior according to grammar 
        A_prior = get_transitions_prior(class2index)
        hmm.A = A_prior
        # trianing 5 times like states in sheet
        for _ in range(5):
            #next:
            # apply HMM learning 
            b_i_t = gmm.get_b_i_t(video)
            alpha_i_t , beta_i_t , gamma_i_t , eta_i_j_t = hmm.learn(video, b_i_t)
                    
            # GMM learning to get new b_i_t
            gmm.train(transcripts, video, gamma_i_t)
            
            # from GMM get new alignment
            b_i_t = gmm.get_b_i_t(video)
            
            # calculate new alignment from new observation model b_i_t 
            alignment = torch.argmax(b_i_t, dim = 0).view(labels.shape[0],)
            
        
        
    return hmm, gmm


def train_HMM_with_MLP(dataloader,model):
    num_subactions = num_states* num_actions
    video1, labels1, _ , transcripts1 =next(iter(dataloader))
    hmm = HMM()
    
    # FOR EACH VIDEO:
    # GET ALIGNMENT
    # DIVIDE ACTIONS TO SUBACTION
    # CREATE #action*#states subactions uniformly
    # create HMM using states and grammar
    # apply HMM learning and GMM
    # from model get new alignments
    # repeat
    for video, labels, _ , transcripts in dataloader:
        video = video[0]
        alignment = get_subaction_alignment(labels.T)
        #initialize probabilies between one action to move forward, but last subaction and first action in each actior according to grammar 
        A_prior = get_transitions_prior(class2index)
        hmm.A = A_prior
        # trianing 5 times like states in sheet
        for _ in range(5):
            #next:
            # apply HMM learning 
            b_i_t = model(video)[0]
            alpha_i_t , beta_i_t , gamma_i_t , eta_i_j_t = hmm.learn(video, b_i_t)
                    
            # train model to get better observation 
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

            for _ in range(1000):

                out = model(video)
                optimizer.zero_grad()
                loss = criterion(out, alignment)
                loss.backward()
                optimizer.step()
            
            # from GMM get new alignment
            b_i_t =model(video)[0]
            
            # calculate new alignment from new observation model b_i_t 
            alignment = torch.argmax(b_i_t, dim = 0).view(labels.shape[0],)
            
        
        
    return hmm, model

def train_HMM_with_RNN(dataloader,model):
    num_subactions = num_states* num_actions
    video1, labels1, _ , transcripts1 =next(iter(dataloader))
    hmm = HMM()
    
    # FOR EACH VIDEO:
    # GET ALIGNMENT
    # DIVIDE ACTIONS TO SUBACTION
    # CREATE #action*#states subactions uniformly
    # create HMM using states and grammar
    # apply HMM learning and GMM
    # from model get new alignments
    # repeat
    for video, labels, _ , transcripts in dataloader:
        video = video[0]
        alignment = get_subaction_alignment(labels.T)
        #initialize probabilies between one action to move forward, but last subaction and first action in each actior according to grammar 
        A_prior = get_transitions_prior(class2index)
        hmm.A = A_prior
        # trianing 5 times like states in sheet
        for _ in range(5):
            #next:
            # apply HMM learning 
            batch = torch.zeros((video.shape[0]-21, 21, video.shape[1]))
            for t in range(10,video.shape[0]):
                patch = video[t-10:t+11]
                batch[t-10] = patch
            b_i_t = model(video)[0]
            alpha_i_t , beta_i_t , gamma_i_t , eta_i_j_t = hmm.learn(video, b_i_t)
                    
            # train model to get better observation 
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
            
            for _ in range(1000):
                # training with 21 frames 
                for t in range(10,video.shape[0]):
                    patch = video[t-10:t+11]
                    out = model(patch)
                    desired_output = alignment[t-10:t+11]
                    optimizer.zero_grad()
                    loss = criterion(out, desired_output)
                    loss.backward()
                    optimizer.step()
            
            # from GMM get new alignment
            b_i_t =model(batch)[0]
            
            # calculate new alignment from new observation model b_i_t 
            alignment = torch.argmax(b_i_t, dim = 0).view(labels.shape[0],)
            
        
        
    return hmm, model
    
num_actions= 12
num_states = 16
batch_size = 1


training_dataset = Dataset(training=True)
class2index = training_dataset.class2index
training_dataloader = torch.utils.data.DataLoader(training_dataset,collate_fn=collate_fn_padd_training,  batch_size=batch_size, shuffle=False, drop_last=False)

test_dataset = Dataset(training=False)
test_dataloader = torch.utils.data.DataLoader(training_dataset,collate_fn=collate_fn_padd_test,  batch_size=batch_size, shuffle=False, drop_last=False)

hmm_with_gmm , gmm = train_HMM_with_GMM(training_dataloader)

simple_MLP = SimpleMLP()
rnn = RNN()

hmm_mlp , trained_mlp = train_HMM_with_model(training_dataloader, simple_MLP)
hmm_mlp , trained_rnn = train_HMM_with_model(training_dataloader, rnn)






