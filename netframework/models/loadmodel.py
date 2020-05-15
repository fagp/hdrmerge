import os
import json
import torch
from torch import nn
from torch.nn import init
from ..utils.utils import Decoder
from ..utils.utils import get_class

def loadmodel(modelname,experimentparams,config_file='defaults/modelconfig.json'):
    model_props = get_model_path(name=modelname, config_file=config_file)

    arch=model_props['arch']
    model_props.pop('arch',None)

    module=model_props['module']
    model_props.pop('module',None)

    for key,value in experimentparams.items():
        model_props[key]=value

    if 'init' in model_props.keys():
        init_type=model_props['init']
        model_props.pop('init',None)
    else:
        init_type='xavier_normal'

    cmodel=get_class(module+'.'+arch)
    net = cmodel(**model_props)

    init_params(net,init_type)

    return net,arch, module

def get_model_path(name, config_file='defaults/modelconfig.json'):
    model_config = json.load(open(config_file),cls=Decoder)
    if name=='':
        name=list(model_config.keys())[0]
    if name not in model_config:
        raise Exception('Model '+name+' not found in '+config_file)
    return model_config[name]

def init_params(net,init_type):
    '''Init layer parameters.'''
    if init_type=='xavier_normal':
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    elif init_type=='xavier_uniform':
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.uniform_(m.weight, 0, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    elif os.path.exists(init_type):
        checkpoint = torch.load(init_type,map_location='cpu')
        net.load_state_dict(checkpoint['net'])
    else:
        raise Exception('Initialization type ' +init_type+ ' not found')
