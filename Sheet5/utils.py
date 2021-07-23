import numpy as np
import torch 



"""

THIS IS A FILE FOR TRYING OUT FUNCTIONS, SAVING FUNCTIONS THAT MIGHT BE USEFUL LATER, and to store functions that are helpful but are not core components 

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
    
        
              