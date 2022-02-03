from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import types
import functools
import logging

class BioModule(torch.jit.ScriptModule):
    @staticmethod
    def get_convert_to_bionet_converter(grad_scale=0.16, dampening_factor=0.6, crystal_thresh=4.5e-05,purge_distance=8.0, accum_neurons: int=0, out_feats=10):
        def converter(mod):
            def bioModulize(self): ## obsolete function
                self.apply(BioModule._bioModulize(grad_scale,dampening_factor,crystal_thresh,purge_distance,accum_neurons))
            
            def _purge(mod):
                if hasattr(mod, 'bio_mod'):
                    mod.bio_mod.purge()
            
            def purge(self):
                self.apply(_purge)
            
            def _crystallize(mod):
                if hasattr(mod, 'bio_mod'):
                    mod.bio_mod.crystallize()
            
            def crystallize(self):
                self.apply(_crystallize)
                
            def _scale_grad(mod):
                if hasattr(mod, 'bio_mod'):
                    mod.bio_mod.scale_grad()
                
            def scale_grad(self):
                self.apply(_scale_grad)

            def wrap_init(func):
                @functools.wraps(func)
                def wrapped_init(self, *args,**kwargs):
                    func(self,*args,**kwargs)
                    self.apply(BioModule._bioModulize(grad_scale,dampening_factor,crystal_thresh,purge_distance,accum_neurons,out_feats))
                return wrapped_init

            mod.__init__ = wrap_init(mod.__init__)
            mod.bioModulize=bioModulize
            mod.purge = purge
            mod.crystallize = crystallize
            mod.scale_grad = scale_grad
            return mod
        return converter
    
    @staticmethod
    def mark_skip_accum(mod):
        mod.accum_skip=True
        return mod
    
    @staticmethod
    def get_parent_callback(mod):
        def __parent():
            return mod
        return __parent
    
    @staticmethod
    def add_accum_step(accum_func, forw_func): 
        @functools.wraps(forw_func)
        def forward_wrapper(*args, **kwargs):
            return accum_func(forw_func(*args, **kwargs))
        return forward_wrapper
    
    @staticmethod
    def _bioModulize(grad_scale, dampening_factor, crystal_thresh, purge_distance, accum_neurons: int=0, out_feats: int=10):
        accum_neurons = int(accum_neurons)
        def __bioModulize(mod):
            if(isinstance(mod,nn.Linear)) and not hasattr(mod, 'bio_mod'):                
                mod.add_module('bio_mod',BioModule(BioModule.get_parent_callback(mod), grad_scale, dampening_factor, crystal_thresh, purge_distance, accum_neurons))
                if accum_neurons>1 and mod.out_features%accum_neurons==0 and not hasattr(mod, 'accum_skip') and not out_feats==mod.out_features:
                    def accum_func(x):
                        return torch.repeat_interleave(x.view(-1,x.size(1)//accum_neurons,accum_neurons).sum(2),accum_neurons,1)
                    mod.forward  = BioModule.add_accum_step(accum_func, mod.forward)
            elif(isinstance(mod,nn.Conv2d)) and not hasattr(mod, 'bio_mod'):                
                mod.add_module('bio_mod',BioModule(BioModule.get_parent_callback(mod), grad_scale, dampening_factor, crystal_thresh, purge_distance, accum_neurons))
                if accum_neurons>1 and mod.out_channels%accum_neurons==0 and not hasattr(mod, 'accum_skip'):                     
                    def accum_func(x):
                        return torch.repeat_interleave(x.view(-1,x.size(1)//accum_neurons,accum_neurons,x.size(-2),x.size(-1)).sum(2),accum_neurons,1)
                    mod.forward  = BioModule.add_accum_step(accum_func, mod.forward)
            elif hasattr(mod, 'bio_mod'):
                mod.bio_mod=BioModule(BioModule.get_parent_callback(mod), grad_scale, dampening_factor, crystal_thresh, purge_distance, accum_neurons)                
        return __bioModulize
        
    def purge(self):
        with torch.no_grad():
            weight = self.parent().weight.data
            try:
                mean, maxi = .0, weight.abs().max()
                purge_threshold = torch.normal(mean,maxi/self.purge_distance+1e-13, weight.shape).to(weight.device).abs()
            except:
                print(maxi)
            rand_w = (torch.rand_like(weight).to(weight.device)*maxi-maxi/2.)*2            
            weight[purge_threshold > weight.abs()] = rand_w[purge_threshold > weight.abs()]
            logging.info((purge_threshold > weight.abs()).sum().item())
            
            self.parent().weight.data = weight
            self.parent().weight.data = weight
            
    def crystallize(self):
        with torch.no_grad():
            weight = self.parent().weight.data.abs()
            grad = self.parent().weight.grad.abs()
            mean = weight.mean()
            idx = ((grad/weight) < self.crystal_thresh) + (grad > mean)
            self.grad_scale[idx]=self.grad_scale[idx]*self.dampening_factor

    def scale_grad(self):
        self.parent().weight.grad *= self.grad_scale 

    def l1_reg(self):
        with torch.no_grad():
            all_linear1_params = torch.cat([x.view(-1) for x in self.parameters()])
            l1_regularization =  torch.norm(all_linear1_params, 1)
        return l1_regularization    
    
    def __init__(self, parent, grad_scale=0.16, dampening_factor=0.6, crystal_thresh=4.5e-05,purge_distance=8.0, accum_neurons: int=0):
        super(BioModule, self).__init__()        
        self.parent=parent
        self.purge_distance=purge_distance
        self.dampening_factor=dampening_factor
        self.crystal_thresh=crystal_thresh
        self.accum_neurons=int(accum_neurons)
        self.grad_scale_param = grad_scale
        self.grad_scale = torch.ones_like(self.parent().weight.data, dtype=float, requires_grad=False) if grad_scale is None else nn.Parameter(1.+(2.*torch.rand_like(self.parent().weight.data,requires_grad=False)*self.grad_scale_param-self.grad_scale_param))
        
    