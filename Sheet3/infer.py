import torch


#get model output
#applies regression

def infer(model, frame):
    # assumption: output is a tensor of shape (N,6) where N is batchsize, 
    # 6 is divided as: 0-3 bbox, 4 actioness score, 5 action label
    # bbox : (x,y,w,h)
    output = model(frame)
    output_T = output.T
    highest_score_index = torch.argmax(output_T[4])
    prediction = output[highest_score_index]
    # apply regression
    
    