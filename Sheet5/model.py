import numpy as np
import torch
import torch.nn as nn 
import sys

'''
classes to implement:
    - HMM
    - GMM
    - simple MLP
    - RNN
    
Note: OUTPUTS OF GMM, MLP, RNN: P(S|X_t)

We use to calc P(X_t|s) using Bayes: P(X_t|s) = P(s|X_t) . const / P(s)
(Observation model for HMM)

'''



#Initialized for each action, each action in video has its own HMM instance
class HMM():
    def __init__(self, num_action=12, num_substates=16):
        # HMM can be defined as a tuple of
        # pi: a vector of length n_states, probabiliy of starting in a certain state
        # A: probability of transition between two states
        # An observation model (GMM, MLP, RNN) will be trained in parallel and used to calculate b
        n_states = num_action * num_substates
        self.n_states = n_states
        # Tensor with 21 Frames
        self.pi = torch.zeros((num_action, num_substates))
        #For our case we always start at state 0, transitions only to same state or the right
        self.pi[:,0] = torch.ones((num_action)) #you can start only at starting subaction of each action
        self.pi = self.pi.reshape((n_states,)) / num_action # normalize
        
        #should have non-zero elements only on (i,i) and (i,i+1)
        # initialize transition priors with 0.8 returning to state 0.2 to next state as suggested in lecture 
        
        #self.A transition matrix gets assigned from outside function get_transitions_prior according to grammar 
        
        
    #source with good explanation:
    #https://www.youtube.com/watch?v=gYma8Gw38Os
    # b_i generated in GMM method
    def get_alpha(self, frames, b_i_t):
        alpha_i_t = torch.zeros((self.n_states, frames.shape[-2]))
        #base case
        alpha_i_t [:,0] = self.pi * b_i_t[:,0]
        print("getting alpha")
        for t in range(frames.shape[-2]-1):
            for j in range(self.n_states):
                Sum=0
                for i in range(self.n_states):
                    Sum+= alpha_i_t[i][t]* self.A[i][j]
                alpha_i_t[j][t+1]= Sum * b_i_t[j][t+1]
                
        
        self.alpha_i_t = alpha_i_t
        return alpha_i_t
    
    #source with good explanation:
    #https://www.youtube.com/watch?v=gYma8Gw38Os
    # b_i generated in GMM method
    def get_beta(self, frames, b_i_t):
        T = frames.shape[-2]
        beta_i_t = torch.zeros((self.n_states, T))
        #base case
        beta_i_t [:,T-1] = torch.ones((1,self.n_states))
        print("getting beta")

        for i in range(frames.shape[-2]-1):
            t = T - i -1
            for j in range(self.n_states):
                Sum=0
                for k in range(self.n_states):
                    Sum+= beta_i_t[k][t]* self.A[k][j]
                beta_i_t[j][t-1]= Sum * b_i_t[j][t-1]
                
        
        self.beta_i_t = beta_i_t
        return beta_i_t
    
    # slide 44 lecture set 5
    def get_gamma(self,frames , b_i_t):
        T = frames.shape[-2]
        gamma_i_t = torch.zeros((self.n_states, T))
        alpha_i_t = self.alpha_i_t
        beta_i_t = self.beta_i_t
        print("getting gamma")

        for t in range(T):
            for i in range (self.n_states):
                gamma_i_t[i][t] = alpha_i_t[i][t] * beta_i_t[i][t]
        
            #normalize
            gamma_i_t[:,t]/= torch.sum(gamma_i_t[:,t])
        self.gamma_i_t = gamma_i_t
        return gamma_i_t
    
    # slide 44 lecture set 5
    def get_eta(self,frames , b_i_t):
        T = frames.shape[-2]
        eta_i_j_t = torch.zeros((self.n_states,self.n_states, T))
        alpha_i_t = self.alpha_i_t
        beta_i_t = self.beta_i_t
        print("getting eta")
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    eta_i_j_t[i][j][t] = alpha_i_t[i][t] *self.A[i][j] * b_i_t[j][t+1] * beta_i_t[j][t+1]
            
            # normalize
            eta_i_j_t[:,:,t]/= torch.sum(eta_i_j_t[:,:,t])
        
        self.eta_i_j_t = eta_i_j_t
        return eta_i_j_t
    
    #slide 48
    def update_pi(self):
        self.pi = self.gamma_i_t[:,0]
    
    #slide 45
    def update_A(self, frames):
        T = frames.shape[-2]
        gamma_i_t = self.gamma_i_t
        eta_i_j_t = self.eta_i_j_t
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.A[i][j] = torch.sum(eta_i_j_t[i,j,:]) / torch.sum(gamma_i_t[i,:])
                
                
    def learn(self, frames, b_i_t):
        print("HMM is learning")
        alpha_i_t = self.get_alpha( frames, b_i_t)
        beta_i_t = self.get_beta( frames, b_i_t) 
        gamma_i_t = self.get_gamma(frames, b_i_t)
        eta_i_j_t = self.get_eta(frames, b_i_t)
        self.update_pi()
        self.update_A(frames)
        
        return alpha_i_t , beta_i_t , gamma_i_t , eta_i_j_t         
        
      
    

class GMM():
    def __init__(self, frames, state_dim = 192, M=16, observation_dim = 64):
            self.state_dim = state_dim
            self.M = M
            self.observation_dim = observation_dim
            
            # init means with random frames 
            self.means_j_l = torch.zeros((state_dim,M,observation_dim))
            for j in range(state_dim):
                for l in range(M):
                    random_index = torch.randint(low = 0, high= frames.shape[-2] , size =(1,))
                    self.means_j_l[j][l]= frames[random_index]
            # init covariances with unit matrices
            self.covariances_j_l = torch.zeros((state_dim,M,observation_dim, observation_dim))
            for j in range(state_dim):
                for l in range(M):
                    self.covariances_j_l[j][l] = torch.eye(observation_dim)
            
            # uniform initlization of weights row-wise 
            self.weights_j_l = torch.ones((state_dim, M)) / M # c in lecture notation
        
    
    # input: list of alignment as ground truth of S
    #list of frames as observations
    # gamma calculated by Baum-Welch algorithm from HMM  of shape [state_dim, T(#nframes)]
    def train(self, alignment, frames, gamma):
        for state in range(self.state_dim):
            # get indices of all features belonging to this state
            indices = alignment==state
            # get features of this state
            state_features = frames[indices]
            b_i = self.get_b_i_t(frames) #current prediction according to current state of model
            # We need b_i_l 
            b_i_l = self.get_b_j_l_t(frames)

            # Formula slide 46
            gamma_i_l_t = gamma *self.weights_j_l * b_i_l / b_i

            # FROM GAMMA I,L GENERATE MEAN , COV, WEIGHTS slide 47
            
            #update means, weights, covariances using gamma_i_l
            for i in range(self.state_dim):
                for l in range(self.M):
                    self.weights_j_l[i][l] = torch.sum(gamma_i_l_t[i][l]) / torch.sum(gamma[i])
                    mean = torch.zeros((1,self.observation_dim))

                    for t in range(frames.shape[-2]):
                        mean+= gamma_i_l_t[i][l][t] * frames[t]
    
                    mean /= torch.sum(gamma_i_l_t[i][l])
                    self.means_j_l[i][l] = mean
                    cov = torch.zeros((self.observation_dim,self.observation_dim))
                    for t in range(frames.shape[-2]):
                        cov+= gamma_i_l_t[i][l][t] * (frames[t] - self.means_j_l[i][l]).T@(frames[t] - self.means_j_l[i][l])
                    cov/=  torch.sum(gamma_i_l_t[i][l])
                    self.covariances_j_l[i][l]= cov
            
        
            
    
    # returns Tensor of shape [#frames, state_dim] where rows are normalised probability distribution (vector size state_dim) for observation over states resulting from the gaussian mixtures
    
    # based on slide 46 action segmentation lecture
    
    def get_b_i_t(self,frames):
        result = torch.zeros((frames.shape[0], self.state_dim))
        b_j_l = self.get_b_j_l_t(frames)
        for idx, frame in enumerate(frames):
            for j in range(self.state_dim): 
                result[idx][j] = torch.sum(self.weights_j_l * b_j_l[j])
                
        return result.T
                
        
    
    # return Tensor with shape (#frames,state_dim, M ) with normalized rows, each representing predictions according only to component l \in M
    def get_b_j_l_t(self,frames):
        result = torch.zeros((frames.shape[0],  self.state_dim, self.M))
        for idx, frame in enumerate(frames):
            for j in range(self.state_dim):
                for l in range(self.M):
                    mean = self.means_j_l[j][l]
                    cov = self.covariances_j_l[j][l]
                    distr = torch.distributions.multivariate_normal.MultivariateNormal(mean,cov)
                    delta = 0.000001 * torch.ones(frame.shape)
                    probability = 10**distr.log_prob(frame+delta) - 10**distr.log_prob(frame-delta)
                    result[idx][j][l]= probability
                    
        return result
       
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim = 64, hidden_dim = 64, out_dim = 192):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax()
        
        
    def forward(self,frame):
        out = self.fc1(frame)
        out = self.activation(out)
        out = self.fc2(out)
        return self.softmax(out)
        
    
class RNN(torch.nn.Module):
    def __init__(self, input_dim = 64, hidden_dim = 256, out_dim = 192, nlayers = 21):
        super(RNN, self).__init__()
        self.GRU = torch.nn.GRU(input_dim, hidden_dim, nlayers)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.ReLU()
        
        
        
    def forward(self,frame, h):
        out, h = self.GRU(frame, h)
        out = self.fc(self.activation(out[:,-1]))
        return out, h
            
        