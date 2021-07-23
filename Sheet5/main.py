import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import Dataset
from model import HMM , GMM , SimpleMLP , RNN
from utils import collate_fn_padd_training, collate_fn_padd_test , get_subaction_alignment , get_transitions_prior

"""In case of any confusion, please check the extended report in the pdf file""" 

    
# given P(s|x) return p(x|s) using equation 5 from paper by Bayes
# model output Tensor of shape [T, num_subactions] describing probability distribution of state probability at each time t (rows are normalized)
# hmm used to get gamma and from it p(s)
def get_p_x_s(model_output,hmm):
    T = model_output.shape[0]
    num_subactions = num_actions * num_states
    # slide 45 lecture set 5
    expected_times_s_comes = torch.sum(hmm.gamma_i_t,dim = 1)
    p_s = expected_times_s_comes / num_subactions # normalize for probability
    P_x_s = torch.zeros(T,num_subactions)
    # Equatipn 5 in paper using Bayes 
    for t in range(T):
        P_x_s[t] = model_output[t] / p_s
        P_x_s[t]/= torch.sum(P_x_s[t]) # normalize
        
    return P_x_s


# given P(s|x) apply equation 10 using HMM priorities, p(s) computed from gamma
# new alignemnt is computed using equation 10 from paper
def get_new_alignment(model_output,hmm ):
    T = model_output.shape[0]
    num_subactions = num_actions * num_states
    p_x_s = get_p_x_s(model_output,hmm)
    new_alignment = torch.zeros((T))
    s_hat_t = torch.zeros((num_subactions))
    for t in range(1,T):
        
        s_t_minus1 = torch.argmax(model_output[t-1])
        s_t = torch.argmax(model_output[t])
        s_hat_t = torch.prod(p_x_s[t] * hmm.A[s_t_minus1][s_t] )
        new_alignment[t] = torch.argmax(s_hat_t)
    
    return new_alignment
        
    
        
    
    
    
        
    

def train_HMM_with_GMM(dataloader):
    print("Training with GMM")
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
        A_prior , _ = get_transitions_prior(class2index)
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
            alignment = get_new_alignment(b_i_t,hmm)
            
        
        
    return hmm, gmm


def train_HMM_with_MLP(dataloader,model):
    print("Training with MLP")
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
        A_prior , _ = get_transitions_prior(class2index)
        hmm.A = A_prior
        # trianing 5 times like states in sheet
        for _ in range(5):
            #next:
            # apply HMM learning 
            b_i_t = model(video).T
            alpha_i_t , beta_i_t , gamma_i_t , eta_i_j_t = hmm.learn(video, b_i_t)
                    
            # train model to get better observation 
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

            for _ in range(1000):
                print("hello")
                out = model(video)
                optimizer.zero_grad()
                loss = criterion(out, alignment)
                loss.backward()
                optimizer.step()
            
            # from GMM get new alignment
            b_i_t =model(video).T
            
            # calculate new alignment from new observation model b_i_t 
            alignment = get_new_alignment(b_i_t,hmm)
            
        
        
    return hmm, model

def train_HMM_with_RNN(dataloader,model):
    print("Training with RNN")

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
        A_prior , _= get_transitions_prior(class2index)
        hmm.A = A_prior
        # trianing 5 times like states in sheet
        for _ in range(5):
            #next:
            # apply HMM learning 
            batch = torch.zeros((video.shape[0]-21, 21, video.shape[1]))
            for t in range(10,video.shape[0]-11):
                patch = video[t-10:t+11]
                batch[t-21] = patch
            h = torch.zeros((21,21,256))
            print(batch[0][None,...].shape,h.shape)
            b_i_t = torch.zeros((192,21))
            for t in range(21):
                res , h = model(batch[t][None,...],h)
                b_i_t[:,t] = res
                #apply learning for each batch individually 
                alpha_i_t , beta_i_t , gamma_i_t , eta_i_j_t = hmm.learn(batch[t], b_i_t)

            
            
                    
            # train model to get better observation 
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
            
            for _ in range(1000):
                # training with 21 frames 
                for t in range(10,video.shape[0]):
                    patch = video[t-10:t+11]
                    print(patch)
                    out = model(patch)
                    print(out)
                    
                    desired_output = alignment[t-10:t+11]
                    optimizer.zero_grad()
                    loss = criterion(out, desired_output)
                    loss.backward()
                    optimizer.step()
            
            # from GMM get new alignment
            b_i_t , h =model(batch)
            b_i_t = b_i_t.T
            
            # calculate new alignment from new observation model b_i_t 
            alignment = get_new_alignment(b_i_t,hmm)
            
            
        
        
    return hmm, model
  
#  equation 6 from the paper is used to calculate the action sequence but does not work for the alignment
# this is why we simply use the output training and generating the sequence same way from equation 10 
def infer_gmm(video, hmm, gmm):
    b_i_t = gmm.get_b_i_t(video)
    alignment = get_new_alignment(b_i_t,hmm)
    return alignment//num_actions # to return to actions instead of subactions 
    
def infer_mlp(video, hmm, mlp):
    b_i_t =mlp(video).T
    alignment = get_new_alignment(b_i_t,hmm)
    return alignment//num_actions

def infer_rnn(video, hmm, rnn):
    b_i_t , h =rnn(video)
    b_i_t = b_i_t.T
    alignment = get_new_alignment(b_i_t,hmm)
    return alignment//num_actions

    


def evaluate_gmm(dataloader, hmm, gmm):
    accuracy = 0
    for video, labels, _ in dataloader:
        video = video[0]
        alignemnt = infer_gmm(video, hmm, gmm)
        accuracy += torch.sum(alignemnt == labels[0])
    return accuracy / len(dataloader)   
        
def evaluate_mlp(dataloader, hmm, mlp):
    accuracy = 0
    for video, labels, _ in dataloader:
        video = video[0]
        alignemnt = infer_mlp(video, hmm, mlp)
        accuracy += torch.sum(alignemnt == labels[0])
    return accuracy / len(dataloader) 

def evaluate_rnn(dataloader, hmm, rnn):
    accuracy = 0
    for video, labels, _ in dataloader:
        video = video[0]
        alignemnt = infer_rnn(video, hmm, rnn)
        accuracy += torch.sum(alignemnt == labels[0])
    return accuracy / len(dataloader) 
    
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

hmm_mlp , trained_mlp = train_HMM_with_MLP(training_dataloader, simple_MLP)
hmm_rnn , trained_rnn = train_HMM_with_RNN(training_dataloader, rnn)

print("accuracy of GMM version: ", evaluate_gmm(test_dataloader,hmm_with_gmm,gmm))

print("accuracy of MLP version: ", evaluate_mlp(test_dataloader,hmm_mlp,trained_mlp))


print("accuracy of MLP version: ", evaluate_mlp(test_dataloader,hmm_rnn,trained_rnn))






