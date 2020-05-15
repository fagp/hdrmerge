import os
import torch
import torchvision
import math
import numpy as np
import scipy.misc as m
from scipy import ndimage
import skimage.color as skcolor
import skimage.util as skutl
from scipy.interpolate import griddata
from skimage.transform import rotate
from skimage.transform import resize
from torch.utils import data
import time
import itertools
from torch.autograd import Variable


#########################################################################################################
class ToTensor(object):

    def __call__(self, sample):
        image=sample['burst']
        gt=sample['gt']

        if len(gt.shape)==2:
            gt = gt[:,:,np.newaxis]
        gt = np.array((gt).transpose((2, 0, 1)))

        image = np.array((image).transpose((2, 0, 1)))
      
        return {'burst': torch.from_numpy(image).float(),'gt': torch.from_numpy(gt).float() } 
#########################################################################################################
class RandomCrop(object):
    '''
    Example of random crop transformation
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image=sample['burst']
        gt=sample['gt']
        

        h, w = gt.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,left: left + new_w,:]
        gt = gt[top: top + new_h,left: left + new_w]

        return {'burst':image,'gt':gt}

