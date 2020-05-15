import torch

def selectschedule(schedulename,optimizer):   
    if schedulename == 'rop':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=20,verbose=1,factor=0.05)
    elif schedulename == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1 )
    elif schedulename == 'exp':
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99 )
    else:
        scheduler = emptyschedule()

    return scheduler

class emptyschedule(object):
    def __init__(self):
        pass
        
    def step(self, metrics, epoch):
        pass