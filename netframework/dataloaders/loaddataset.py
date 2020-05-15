import json
import numpy as np
from ..utils.utils import Decoder
from ..utils.utils import get_class
from importlib import import_module
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def loaddataset(datasetname,experimentparam,batch_size=1,worker=1,config_file='defaults/dataconfig_train.json'):     
    #load dataset configuration (json)
    data_props = get_data_path(name=datasetname,config_file=config_file)
    module=data_props['module']
    data_props.pop('module',None)

    for key,value in experimentparam.items():
        data_props[key]=value

    if 'transform_param' in data_props:
        transformstr=data_props['transform_param']
        data_props.pop('transform_param',None)
    else:
        raise 'Please define a default transform \'transform_param\' behavior in '+config_file

    #setup transforms
    tr = import_module( module+'.ctransforms' )
    transformlist=transformstr.replace(' ','').split('),')
    transformstr=''
    for transf in transformlist:
        transformstr += 'tr.'+transf+'),'
    transformstr=transformstr[:-2]

    transform = eval('transforms.Compose(['+transformstr+'])')

    cdataset=get_class(module+'.dataset.cdataset')

    #dataset 
    ddatasets = cdataset(**data_props,transform_param=transform)

    #loader
    tsampler = SubsetRandomSampler(np.random.permutation(len(ddatasets)))
    dloader = DataLoader(ddatasets, batch_size=batch_size, sampler=tsampler, num_workers=worker)

    return ddatasets, dloader, module


def get_data_path(name, config_file='defaults/dataconfig_train.json'):
    data = json.load(open(config_file),cls=Decoder)
    if name=='':
        name=list(data.keys())[0]
    if name not in data:
        raise 'Dataset '+name+' not found in '+config_file
    return data[name]
