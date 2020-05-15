import os
import json
import torch
import numpy as np
from torch import nn
from torch.nn import init
from torch.autograd import Variable

def parse_cuda(args):
    if args.use_cuda>-1:
        print('Using CUDA')
        device = torch.device('cuda:'+str(args.use_cuda))
        use_parallel=args.parallel
    else:
        print('Using CPU')
        device = torch.device('cpu')
        use_parallel=False
    
    return device, use_parallel

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()        
    def reset(self):
        self.array = []
        self.val=0
        self.total_avg = 0
        self.total_sum = 0
        self.total_count = 0

        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def new_local(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val=val
        self.array= self.array + [val]
        
        self.total_sum += val * n
        self.total_count += n
        if self.total_count>50:
            self.total_avg = np.median(np.array(self.array[-50:]))#self.total_sum / self.total_count
        else:
            self.total_avg = self.total_sum / self.total_count

        self.sum += val * n
        self.count += n
        self.avg = self.array[-1]
    
    def load(self,narray,n=1):
        for val in narray:
            self.update(val,n)

class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                try:
                    return float(o)
                except ValueError:
                    return self.str_to_bool(o)
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o
    
    def str_to_bool(self,s):
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            return s

def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m
