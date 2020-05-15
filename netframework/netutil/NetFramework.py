# import gc
import os
# import sys
import json
import time
# import torch
# import random
import signal
import argparse
# import numpy as np
# import torch.nn as nn
from  visdom import Visdom
from ..utils.utils import *
# from scipy.misc import imsave
import torch.nn.functional as F
from ..utils.utils import Decoder
# import torchvision.models as models
# from torch.autograd import Variable
from importlib import import_module
from ..utils import graphics as gph
import torch.backends.cudnn as cudnn
from ..models.loadmodel import loadmodel
from ..loss.selectloss import selectloss
from ..loss.selectloss import get_metric_path
from ..dataloaders.loaddataset import loaddataset
from ..optimizers.selectopt import selectoptimizer
from ..optimizers.selectschedule import selectschedule
import warnings
import random

warnings.filterwarnings("ignore")

class NetFramework():
    def __init__(self, defaults_path):

        parser = argparse.ArgumentParser(description='Net framework arguments description')
        parser.add_argument('--experiment', nargs='?', type=str, default='experiment', help='Experiment name')
        parser.add_argument('--model', nargs='?', type=str, default='', help='Architecture to use')
        parser.add_argument('--modelparam', type=str, default='{}', help='Experiment model parameters')
        parser.add_argument('--dataset', nargs='?', type=str, default='', help='Dataset key specified in dataconfig_*.json')
        parser.add_argument('--datasetparam', type=str, default='{}', help='Experiment dataset parameters')
        parser.add_argument('--imsize', nargs='?', type=int, default=200, help='Image resize parameter')

        parser.add_argument('--visdom', action='store_true', help='If included shows visdom visulaization')
        parser.add_argument('--show_rate', nargs='?',type=int, default=4, help='Visdom show after num of iterations (used with --visdom)')
        parser.add_argument('--print_rate', nargs='?',type=int, default=4, help='Print after num of iterations')
        parser.add_argument('--save_rate', nargs='?',type=int, default=10, help='Save after num of iterations (if --save_rate=0 then no save is done during training)')

        parser.add_argument('--use_cuda', nargs='?',type=int, default=0, help='GPU device (if --use_cuda=-1 then CPU used)')
        parser.add_argument('--parallel', action='store_true', help='Use multiples GPU (used only if --use_cuda>-1)')
        parser.add_argument('--epochs', nargs='?', type=int, default=1000, help='Number of epochs')
        parser.add_argument('--batch_size', nargs='?', type=int, default=1, help='Minibatch size')
        parser.add_argument('--batch_acc', nargs='?', type=int, default=1, help='Minibatch accumulation')
        parser.add_argument('--train_worker', nargs='?', type=int, default=1, help='Number of training workers')
        parser.add_argument('--test_worker', nargs='?', type=int, default=1, help='Number of testing workers')

        parser.add_argument('--optimizer', nargs='?', type=str, default='RMSprop', help='Optimizer to use')
        parser.add_argument('--optimizerparam', type=str, default='{}', help='Experiment optimizer parameters')
        parser.add_argument('--lrschedule', nargs='?', type=str, default='none', help='LR Schedule to use')
        parser.add_argument('--loss', nargs='?', type=str, default='', help='Loss function to use')
        parser.add_argument('--lossparam', type=str, default='{}', help='Loss function parameters')
        parser.add_argument('--resume', action='store_true', help='Resume training')

        parser.add_argument('--seed', nargs='?',type=int, default=123, help='Random seed (for reproducibility)')

        args = parser.parse_args()

        if args.seed!=-1:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        # create outputs folders
        root='../out'
        experimentpath=(os.path.join(root,args.experiment))
        folders={ 'root_path':root, 'experiment_path':experimentpath, 'model_path':os.path.join(experimentpath,'model'), 'images_path':os.path.join(experimentpath,'images') }

        for i in range(2):
            for folder, path in folders.items():
                if not os.path.isdir(path):
                    try:
                        os.mkdir(path)  
                    except:
                        pass

        json.dump(vars(args),open(os.path.join(experimentpath,'args.json'),'w'))
        args.folders=folders

        args.lossparam=json.loads(args.lossparam.replace("'","\""),cls=Decoder)
        args.datasetparam=json.loads(args.datasetparam.replace("'","\""),cls=Decoder)
        args.modelparam=json.loads(args.modelparam.replace("'","\""),cls=Decoder)
        args.optimizerparam=json.loads(args.optimizerparam.replace("'","\""),cls=Decoder)

        # Parse use cuda
        self.device, self.use_parallel = parse_cuda(args)
        torch.cuda.set_device(args.use_cuda)

        # Visdom visualization
        self.visdom=args.visdom
        if self.visdom==True:
            self.vis = Visdom(use_incoming_socket=False)
            self.vis.close(env=args.experiment)
            self.visplotter = gph.VisdomLinePlotter(self.vis, env_name=args.experiment)
            self.visheatmap = gph.HeatMapVisdom(self.vis, env_name=args.experiment)
            self.visimshow  = gph.ImageVisdom(self.vis, env_name=args.experiment)
            self.vistext    = gph.TextVisdom(self.vis, env_name=args.experiment)

        # Showing results rate
        self.print_rate = args.print_rate
        self.show_rate = args.show_rate
        self.save_rate = args.save_rate
        
        self.init_epoch=0
        self.current_epoch=0
        self.epochs=args.epochs
        self.folders=args.folders
        self.bestmetric=0
        self.batch_size=args.batch_size
        self.batch_acc=args.batch_acc

        # Load datasets
        print('Loading dataset: ',args.dataset)
        self.traindataset,self.train_loader, self.dmodule = loaddataset(datasetname=args.dataset,
                                        experimentparam=args.datasetparam,
                                        batch_size=args.batch_size,
                                        worker=args.train_worker,
                                        config_file=os.path.join(defaults_path,'dataconfig_train.json'))
        
        self.testdataset,self.test_loader,_ = loaddataset(datasetname=args.dataset,
                                        experimentparam=args.datasetparam,
                                        batch_size=args.batch_size,
                                        worker=args.test_worker,
                                        config_file=os.path.join(defaults_path,'dataconfig_test.json'))

        self.warp_var_mod = import_module( self.dmodule+'.dataset' )

        # Setup model
        print('Loading model: ',args.model)
        self.net, self.arch, self.mmodule = loadmodel(modelname=args.model,
                                        experimentparams=args.modelparam,
                                        config_file=os.path.join(defaults_path,'modelconfig.json'))

        self.net.to(self.device)
        if self.use_parallel:
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True


        # Setup Optimizer
        print('Selecting optimizer: ',args.optimizer)
        self.optimizer = selectoptimizer(args.optimizer,self.net,args.optimizerparam)
        
        # Setup Learning Rate Scheduling
        print('LR Schedule: ',args.lrschedule)
        self.scheduler = selectschedule(args.lrschedule, self.optimizer)

        # Setup Loss criterion
        print('Selecting loss function: ',args.loss)
        self.criterion, self.losseval = selectloss(lossname=args.loss,
                                        parameter=args.lossparam,
                                        config_file=os.path.join(defaults_path,'loss_definition.json'))
        self.criterion.to(self.device)
        self.trlossavg = AverageMeter()
        self.vdlossavg = AverageMeter()
        
        # Others evaluation metrics
        print('Selecting metrics functions:')
        metrics_dict=get_metric_path(os.path.join(defaults_path,'metrics.json'))
        self.metrics = dict()
        self.metrics_eval = dict()
        self.trmetrics_avg = dict()
        self.vdmetrics_avg = dict()

        for key,value in metrics_dict.items():
            self.metrics[key],self.metrics_eval[key] = selectloss(lossname=value['metric'],
                                        parameter=value['param'],
                                        config_file=os.path.join(defaults_path,'loss_definition.json'))
            self.metrics[key].to(self.device)
            self.trmetrics_avg[key]=AverageMeter()
            self.vdmetrics_avg[key]=AverageMeter()

        if args.resume:
            self.resume()
        
        signal.signal(signal.SIGTERM, self.savemodel)
        self.args=args

    def do_train(self):
        for current_epoch in range(self.init_epoch,self.epochs):
            print('epoch ',current_epoch)
            self.current_epoch=current_epoch
            
            # Forward over validation set
            avgloss, avgmetric=self.validation(current_epoch)
            self.scheduler.step(avgloss, current_epoch)

            # If obtained validation accuracy improvement save network in model/bestmodel.t7
            if self.bestmetric<avgmetric:
                print('Validation metric improvement ({:.3f}) in epoch {} \n'.format(avgmetric,current_epoch))
                self.bestmetric=avgmetric
                self.savemodel(os.path.join(self.folders['model_path'],'bestmodel.t7'))

            save_ = True if self.save_rate!=0 and (current_epoch % self.save_rate)==0 else False
            # Save netowrk after self.save_rate epochs
            if save_:
                print('Saving checkpoint epoch {}\n'.format(current_epoch))
                self.savemodel(os.path.join(self.folders['model_path'],'epoch{}model.t7'.format(current_epoch)))

            # Forward and backward over training set
            self.train(current_epoch)
            self.valid_visualization(current_epoch,3)
        
        # Save last model netowrk
        self.savemodel(os.path.join(self.folders['model_path'],'lastmodel.t7'))

    ## Train function
    def train(self,current_epoch):
        ttime=time.time()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        self.trlossavg.new_local()
        for key,value in self.trmetrics_avg.items():
            self.trmetrics_avg[key].new_local()

        self.net.train()   

        end = time.time()
        total_train=len(self.train_loader)
        for i, sample in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            iteration=float(i)/total_train +current_epoch
            sample = self.warp_var_mod.warp_Variable(sample,self.device)
            images=sample['image']

            outputs = self.net(images)
            kwarg=eval(self.losseval)
            loss=self.criterion(**kwarg)
            loss.backward()
            if (i+1)%self.batch_acc ==0 or (i+1)==total_train:
                self.optimizer.step()

            self.trlossavg.update(loss.item(),images.size(0))
            for key,value in self.metrics_eval.items():
                kwarg=eval(self.metrics_eval[key])
                metric=self.metrics[key](**kwarg)
                self.trmetrics_avg[key].update(metric.item(),images.size(0))

            if (i+1)%self.batch_acc ==0 or (i+1)==total_train:
                self.optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i % self.print_rate)==0:
                strinfo  = '| Train: [{0}][{1}/{2}]\t'                
                strinfo += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'     
                strinfo += 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'

                print(
                        strinfo.format(
                            current_epoch, i+1, total_train,
                            batch_time=batch_time,
                            data_time=data_time
                            )                
                        ,end=''
                        )

                for key,value in self.trmetrics_avg.items():
                    print('{} {:.3f} ({:.3f})\t'.format(key,value.val,value.avg),end='')

                print('loss {:.3f} ({:.3f})'.format(self.trlossavg.val,self.trlossavg.avg))
                

            if self.visdom==True and (  ((i+1) % self.show_rate)==0 or ((i+1) % total_train)==0 ):
                info = {'loss':self.trlossavg}
                
                for key,value in self.trmetrics_avg.items():
                    info[key]=value

                for tag, value in info.items():
                    self.visplotter.show(tag, 'train', iteration, value.avg )
                    self.visplotter.show(tag, 'train_mean', iteration, value.total_avg )

        print('|Total time: {:.3f}'.format(time.time()-ttime))


    def validation(self,current_epoch): 
        ttime=time.time()  
        data_time = AverageMeter()
        batch_time = AverageMeter()

        self.vdlossavg.new_local()
        for key,value in self.vdmetrics_avg.items():
            self.vdmetrics_avg[key].new_local()  

        end = time.time()
        total_valid=len(self.test_loader)
        with torch.no_grad():
            for i, sample in enumerate(self.test_loader):
                data_time.update(time.time() - end)

                iteration=float(i)/total_valid +current_epoch-1
                sample = self.warp_var_mod.warp_Variable(sample,self.device)
                images=sample['image']

                outputs = self.net(images)
                kwarg=eval(self.losseval)
                loss=self.criterion(**kwarg)

                self.vdlossavg.update(loss.item(),images.size(0))
                for key,value in self.metrics_eval.items():
                    kwarg=eval(self.metrics_eval[key])
                    metric=self.metrics[key](**kwarg)
                    self.vdmetrics_avg[key].update(metric.item(),images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i%self.print_rate==0:
                    strinfo  = '| Valid: [{0}][{1}/{2}]\t'                
                    strinfo += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'     
                    strinfo += 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 

                    print(
                            strinfo.format(
                                current_epoch, i+1, total_valid,
                                batch_time=batch_time,
                                data_time=data_time,
                                )
                            ,end=''               
                            )

                    for key,value in self.vdmetrics_avg.items():
                        print('{} {:.3f} ({:.3f})\t'.format(key,value.val,value.avg),end='')

                    print('loss {:.3f} ({:.3f})'.format(self.vdlossavg.val,self.vdlossavg.avg))


                if self.visdom==True and current_epoch!=self.init_epoch and (  ((i+1) % self.show_rate)==0 or ((i+1) % total_valid)==0 ):
                    info = {'loss':self.vdlossavg}

                    for key,value in self.vdmetrics_avg.items():
                        info[key]=value

                    for tag, value in info.items():
                        self.visplotter.show(tag, 'valid', iteration, value.avg )
                        self.visplotter.show(tag, 'valid_mean', iteration, value.total_avg )
        
        if list(self.vdmetrics_avg.keys()):
            watch_metric=self.vdmetrics_avg[list(self.vdmetrics_avg.keys())[0]]
        else:
            watch_metric=self.vdlossavg
        print('|Total time: {:.3f}'.format(time.time()-ttime))

        return self.vdlossavg.avg, watch_metric.avg

    def valid_visualization(self,current_epoch,index=0,save=False):   
        with torch.no_grad(): 
            sample=self.testdataset[ index ]
            sample['image'].unsqueeze_(0)

            sample=self.warp_var_mod.warp_Variable(sample,self.device)
            images=sample['image']
            img=images[0].cpu().numpy()
            if self.visdom==True:
                self.visimshow.show('Image',img)

        return 1


    def savemodel(self,modelpath='',killsignal=None):
        if modelpath=='' or killsignal is not None:
            print('Saving checkpoint epoch {}\n'.format(self.current_epoch))
            modelpath=os.path.join(self.folders['model_path'],'epoch{}model.t7'.format(self.current_epoch))
        to_save= self.net.module if self.use_parallel else self.net
        state = {
                'epoch': self.current_epoch,
                'arch':  self.arch,
                'net':   to_save.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'bestmetric': self.bestmetric
            }
        torch.save(state, modelpath)

        metrics_dict={'train_loss':self.trlossavg,'valid_loss':self.vdlossavg}
        for key,value in self.trmetrics_avg.items():
            metrics_dict['train_'+key]=value
        for key,value in self.vdmetrics_avg.items():
            metrics_dict['valid_'+key]=value

        for tag, value in metrics_dict.items():
            np.savetxt(self.folders['experiment_path']+'/'+tag+'.txt', np.array(value.array) , delimiter=',', fmt='%3.6f') 
        
        if killsignal is not None:
            exit(-1)
    
    def loadmodel(self,modelpath):
        if os.path.isfile(modelpath):
            checkpoint = torch.load(modelpath,map_location='cpu')
            to_load= self.net.module if self.use_parallel else self.net
            to_load.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch=checkpoint['epoch']
            self.arch=checkpoint['arch']
            self.bestmetric=checkpoint['bestmetric']

            files = [ f for f in sorted(os.listdir(self.folders['experiment_path'])) if (f.find('train_')!=-1 and f.find('.txt')!=-1) ]
            for f in files:
                narray=np.loadtxt(os.path.join(self.folders['experiment_path'],f),delimiter=',')
                metric=f[6:f.find('.txt')]
                if metric=='loss':
                    self.trlossavg.load(narray,1)
                if metric in self.trmetrics_avg:
                    self.trmetrics_avg[metric].load(narray.tolist(),1)

            files = [ f for f in sorted(os.listdir(self.folders['experiment_path'])) if (f.find('valid_')!=-1 and f.find('.txt')!=-1) ]
            for f in files:
                narray=np.loadtxt(os.path.join(self.folders['experiment_path'],f),delimiter=',')
                metric=f[6:f.find('.txt')]
                if metric=='loss':
                    self.vdlossavg.load(narray,1)
                if metric in self.vdmetrics_avg:
                    self.vdmetrics_avg[metric].load(narray.tolist(),1)

        else:
            raise Exception('Model not found')

    def resume(self):
        if os.path.isdir(self.folders['model_path']):
            files = [ f for f in sorted(os.listdir(self.folders['model_path'])) if (f.find('epoch')!=-1 and f.find('model.t7')!=-1) ]
            if files:
                self.init_epoch = max([int(f[5:f.find('model.t7')]) for f in files])+1
                self.loadmodel(os.path.join(self.folders['model_path'], 'epoch'+str(self.init_epoch-1)+'model.t7' ))
                print('Resuming on epoch'+str(self.init_epoch-1))

