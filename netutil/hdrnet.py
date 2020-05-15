import torch
from netframework.netutil.NetFramework import NetFramework
import torch.nn.functional as F
import os
import numpy as np
from scipy.io import savemat, loadmat

class HDRNet(NetFramework):
    def __init__(self,default_path):
        NetFramework.__init__(self,default_path)
        pass


    def valid_visualization(self,current_epoch,index=0,save=False):  
        with torch.no_grad():
            index=np.random.randint(0,len(self.testdataset))

            sample=self.testdataset[ index ]
            sample['burst'].unsqueeze_(0)
            sample['gt'].unsqueeze_(0)

            sample=self.warp_var_mod.warp_Variable(sample,self.device)
            burst=sample['image']
            labels=sample['gt']

            outputs = self.net(burst)      

            if self.visdom==True:
                for i in range(outputs.size(1)):
                    self.visheatmap.show('output_ch'+str(i),(outputs[:,i,:,:]).detach().cpu().numpy()[0],'jet',0.5)

                for i in range(labels.size(1)):
                    self.visheatmap.show('label_ch'+str(i),(labels[:,i,:,:]).detach().cpu().numpy()[0],'jet',0.5)

                if outputs.size(1)>1:
                    self.visimshow.show('output',(outputs).detach().cpu().numpy()[0])
                
                if labels.size(1)>1:
                    self.visimshow.show('label',(labels).detach().cpu().numpy()[0])

                for i in range(burst.size(1)):
                     self.visheatmap.show('raw'+str(i),(burst[:,i,:,:]).detach().cpu().numpy()[0],'jet',0.5)
                self.vistext.show('Burst name',self.testdataset.dataprov.getimagename())
        return 1
