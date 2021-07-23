




# function to eval the model
# for each frame: if right prediction, return IOU
# else zero


def eval(test_dataloader, model):
    model.eval()