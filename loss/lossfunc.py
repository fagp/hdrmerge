import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#########################################################################################################    
class NMSE(nn.Module):

    def __init__(self, power=6):
        super(NMSE, self).__init__()
        self.power = power

    def crop(self,w,h,target):
        nt,ht,wt= target.size()
        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2
        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w].clone()
        
        return target


    def forward(self, input, target):
        loss0 = (2/ (torch.exp( -self.power* torch.abs(input - target ))+1 )) -1
        max_input,_= torch.max(torch.max(input,dim=3)[0],dim=2)
        max_target,_= torch.max(torch.max(target,dim=3)[0],dim=2)
        min_input,_= torch.min(torch.min(input,dim=3)[0],dim=2)
        min_target,_= torch.min(torch.min(target,dim=3)[0],dim=2)

        return loss0.mean()+torch.abs(max_input-max_target).max()+torch.abs(min_input-min_target).max()

