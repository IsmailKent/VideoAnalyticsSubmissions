import numpy as np
import torch 



"""

THIS IS A FILE FOR TRYING OUT FUNCTIONS, SAVING FUNCTIONS THAT MIGHT BE USEFUL LATER

"""

#function to calculate covariance of set of features
# source: https://github.com/pytorch/pytorch/issues/19037
def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)





#uses grammar
# viterbi algorithm
# needs the whole alignment
class GMM():
    # We use number of Gaussian components exactly equal to number of states
    def __init__(self, state_dim = 16, M=16, observation_dim = 2048):
            self.state_dim = state_dim
            self.M = M
            self.observation_dim = observation_dim
            self.gmm = GMM(n_components=16)
            
    def train(self,frames):
        self.gmm.fit(frames)
        
    def get_b(self,frames):
        return self.gmm.predict_proba(frames)
    
    
class HMMGaussian():
    def __init__(self,n_states = 16, n_gaussian_components = 16):
        #initizalize with uniform weights 
        weights = torch.ones((n_gaussian_components,)) / n_gaussian_components
        
        # initialize transition priors with 0.8 returning to state 0.2 to next state as suggested in lecture 
        transitions_prior = torch.zeros((n_states,n_states))
        for i in range(n_states):
            transitions_prior[i][i] = 0.8
            if i<n_states-1:
                transitions_prior[i][i+1]=0.2
        # initialize starting state to 1 for first state, always start in state 0
        start_prior = torch.zeros((n_states,))
        start_prior[0]=1
        self.hmm = hmm.GMMHMM(n_components = n_states, n_mix = n_gaussian_components , weights_prior  = weights ,startprob_prior = start_prior,  transmat_prior =transitions_prior , n_iter = 5  )
        
    def train(self, frames):
        self.hmm.fit(frames)
    
    def predict(self, frames):
        return self.hmm.predict(frames)
    
    def get_probs(self, frames):
        return self.hmm.predict_proba(frames)
        

        
"""
class GMM():
    # We use number of Gaussian components exactly equal to number of states
    def __init__(self, state_dim = 16, M=16, observation_dim = 2048):
            self.state_dim = state_dim
            self.M = M
            self.observation_dim = observation_dim
            # TODO: find way to initizale the means and covariances, maybe random elements and unit matrices 
            self.means = torch.zeros((state_dim,observation_dim))
            self.covariances = torch.zeros((state_dim,observation_dim, observation_dim))
            self.weights = torch.ones((M,)) / M # c in lecture notation,initialize uniform
        
    
    # input: list of alignment as ground truth of S
    #list of frames as observations
    # gamma calculated by Baum-Welch algorithm from HMM 
    def train(self, alignment, frames, gamma):
        for state in range(self.state_dim):
            # get indices of all features belonging to this state
            indices = alignment==state
            # get features of this state
            state_features = frames[indices]
            b = get_b(frames) #current prediction according to current state of model
            # We need b_i_l 
            b_i_l = get_i_l(frames)

            gamma_i_l = 
            ## TODO: From GAMMA GENERATE GAMMA I,L SLIDe 46,
            # FROM GAMMA I,L GENERATE MEAN , COV, WEIGHTS slide 47
        
            
            
        pass
    
    # returns Tensor of shape [#frames, state_dim] where rows are normalised probability distribution (vector size state_dim) for observation over states resulting from the gaussian mixtures
    # based on slide 46 action segmentation lecture
    
    def get_b(self,frames):
        # use MixtureSameFamily
        pass
    
    # return Tensor with shape (#frames, M,state_dim) with normalized rows, each representing predictions according only to component l \in M
    def get_i_l(self,frames):
        result = torch.zeros((frames.shape[0], self.M, self.state_dim))
        for idx, frame in enumerate(frames):
            for l in range(self.M):
                distr = torch.distributions.multivariate_normal.MultivariateNormal(self.means[l],self.covariances[l])
                probability = distr.log_prob(frame)
                result[idx][l]= # how to make prob for each state? what is mue jl and sigma jl?
        """
        
num_classes=12      
    # function for zero padding for dataloader because of variable video length, in case of using batch size >1
def collate_fn_padd_training(batch):
        batch_input , batch_target , transcripts = [list(t) for t in zip(*batch)] 
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), max(length_of_sequences),  np.shape(batch_input[0])[1], dtype=torch.float)
        
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        
        mask = torch.zeros(len(batch_input),  max(length_of_sequences), num_classes,dtype=torch.float)
 
        for i in range(len(batch_input)):
            batch_input_tensor[i, :np.shape(batch_input[i])[0],: ] = batch_input[i]
            
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = batch_target[i]
            
            mask[i, :np.shape(batch_target[i])[0], :] = torch.ones( batch_target[i].shape[0], num_classes)
            
        return batch_input_tensor, batch_target_tensor, mask , transcripts

def collate_fn_padd_test(batch):
        batch_input , batch_target , _ = [list(t) for t in zip(*batch)] 
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), max(length_of_sequences),  np.shape(batch_input[0])[1], dtype=torch.float)
        
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        
        mask = torch.zeros(len(batch_input),  max(length_of_sequences), num_classes,dtype=torch.float)
 
        for i in range(len(batch_input)):
            batch_input_tensor[i, :np.shape(batch_input[i])[0],: ] = batch_input[i]
            
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = batch_target[i]
            
            mask[i, :np.shape(batch_target[i])[0], :] = torch.ones( batch_target[i].shape[0], num_classes)
            
        return batch_input_tensor, batch_target_tensor, mask
    
    
    
num_states = 16
num_actions = 12
num_subactions = num_states * num_actions

def get_subaction_alignment(alignment):
    new_alignment = torch.zeros(alignment.shape)
    for i in range(alignment.shape[0]):
        current_action = alignment[i]
        length=1
        for j in range(i,alignment.shape[0]):
            if alignment[j]!=alignment[i] or j==alignment.shape[0]-1:
                i = j
                break
            length+=1
        for k in range(i-length,i):
            new_alignment[k] = current_action * num_states + k//num_states
    return new_alignment

def get_transitions_prior(class2index):
    grammar =[]
    grammar_file = open("./Data/grammar.txt")
    for line in grammar_file:
            line = line.rstrip('\n')
            path =  [int(class2index[x]) for x in line.split(" ")[:-1]]
            grammar.append(path)
    grammar_file.close()
    A_prior = torch.zeros((num_subactions,num_subactions))
    # add transitions within one action
    for i in range(num_subactions):
        A_prior[i][i]=0.8
        if i%num_states!=0:
            if i<num_subactions-1:
                A_prior[i][i+1]=0.2
    # add grammar transitions:
    for path in grammar:
        for i in range(len(path)-1):
            A_prior[path[i]*(num_states+1)-1][[path[i+1]*(num_states+1)-1]] = 0.2
    # normalize rows
    for i in range(A_prior.shape[0]):
        A_prior[i] = A_prior[i] / torch.sum(A_prior[i])
        
    return A_prior, grammar
    
        
              