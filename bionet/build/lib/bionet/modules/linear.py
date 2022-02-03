import torch
from torch import nn
from bionet.biomodule import BioModule as bm

class FCNet(nn.Module):
    def __init__(self, in_feats=784, shapes=[1000], num_classes=10, activation_func_mod=nn.ReLU(inplace=True)):
        super(FCNet, self).__init__()
        self.activation_func_mod = activation_func_mod
        fc1 = nn.Linear(in_feats, shapes[0])        
        layers=[]
        if len(shapes)>1:
            layers = ((nn.Linear(shapes[idx],shapes[idx+1]), activation_func_mod) for idx in range(len(shapes)-1))
            layers = (item for sublist in layers for item in sublist)
        self.fc = nn.Sequential(fc1,activation_func_mod,*layers)      
        self.cl = bm.mark_skip_accum(nn.Linear(shapes[-1],num_classes))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.cl(x)
        return x