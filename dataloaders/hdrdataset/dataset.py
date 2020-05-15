import os
import torch
import random
import torchvision
import numpy as np
import scipy.misc as m
from skimage import filters
from torch.utils import data
import skimage.util as skutl
import skimage.color as skcolor
from torch.autograd import Variable

#for load images. might be used
from .burstutl import *



class cdataset(data.Dataset):
    def __init__(self, path_raw, path_result, ext_raw='dng',ext_res='final.jpg',size_burst=9,transform_param=None,num_burst0=0,num_burst1=100):
        self.transform_param=transform_param
        self.range=[f for f in range(num_burst0,num_burst1)]
        self.dataprov = burstProvide(path_raw, path_result, ext_raw,ext_res,size_burst,self.range)
        self.size_burst=size_burst
        

    def __len__(self):
        return len(self.range)

    def __getitem__(self, index):
        np.random.seed( random.randint(0, 2**32))
        
        hdr_gt,raw_burst,Flag = self.dataprov.getimage(index % len(self.range))   
        while not Flag: #if size mismatch load another image
#            print(self.dataprov.getimagename())
            # self.dataprov.exclude(index % len(self.range))
            # del self.range[index % len(self.range)]            
            index=np.random.randint(0,len(self.range))
            hdr_gt,raw_burst,Flag = self.dataprov.getimage(index % len(self.range))

        
        sample = {'burst': raw_burst, 'gt': hdr_gt} 

        if self.transform_param is not None:
            sample = self.transform_param(sample)

        return sample

def warp_Variable(sample,device):
    images, gt = sample['burst'], sample['gt']
    images=images.to(device)
    gt=gt.to(device)
    sample = {'image': images,'gt': gt}
    return sample
