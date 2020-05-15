import torch

def selectoptimizer(optimizername, net, experimentparams):
    if optimizername=='Adam':
        optimizer = torch.optim.Adam(net.parameters(), **experimentparams)
    elif optimizername=='SGD':
        optimizer = torch.optim.SGD(net.parameters(), **experimentparams)
    elif optimizername=='RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), **experimentparams)
    else:
        raise 'Optimizer {} not found'.format(optimizername)

    return optimizer