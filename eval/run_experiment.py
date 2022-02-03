from bionet.experiment import run_sequencial, run_whole
from bionet.modules.linear import FCNet
from bionet.modules.alexnetMini import AlexNetMini
from bionet.modules.resnetMini import resnet20,resnet32,resnet44,resnet56,resnet110,resnet1202
import torch
import numpy as np
import pickle as pkl
from time import time
import logging
logging.basicConfig(filename='experiment_.log', level=logging.INFO)

def eval_resnet(model_name, num_channels, num_classes):
    if '1202' in model_name:
        return resnet1202(num_channels, num_classes)
    elif '32' in model_name:
        return resnet32(num_channels, num_classes)
    elif '44' in model_name:
        return resnet44(num_channels, num_classes)
    elif '56' in model_name:
        return resnet56(num_channels, num_classes)
    elif '110' in model_name:
        return resnet110(num_channels, num_classes)
    else :
        return resnet20(num_channels, num_classes)
    

def run(exp = 'whole' , model_class = "FCNet", model_args = {'in_feats':784, 'shapes':[1000], 'num_classes':10}, 
        optimizer = 'SGD', optimizer_args= {'lr':1e-3}, grad_scale=0.09, batch_size=100, vbatch_size=1000,
        dampening_factor = 0.6,fold=5, crystal_thresh=4.5e-5, purge_distance=14.0, accum_neurons=2,
       epochs=1, dataset_name='MNIST', purge=True, scale_grad=True, crystallize=False, seq_num=2, verbose=False, cuda_device=None):
    
    str_model_class = model_class
    str_optimizer = optimizer
    
    optimizer = torch.optim.SGD
    num_channels = 3 if dataset_name.upper() != 'MNIST' else 1
    num_classes = 100 if dataset_name.upper()[-3:] == '100' else 10
    
    if model_class == "FCNet":
        
        model_class = FCNet
        if dataset_name=='MNIST':
            model_args = {'in_feats':784, 'shapes':[1000], 'num_classes':10}
        elif "100" in dataset_name:
                model_args = {'in_feats':3*32*32, 'shapes':[3000,3000], 'num_classes':100}
        else:
            model_args = {'in_feats':3*32*32, 'shapes':[3000,1000], 'num_classes':10}
    elif 'resnet' in model_class:
        model_class, model_args =  eval_resnet(model_class, num_channels, num_classes)
    else:
        model_args = {"num_classes":num_classes, "num_chans":num_channels, "target":dataset_name.upper()}
        model_class = AlexNetMini
    
    
    if exp == "whole":
        acc, res = run_whole(model_class =  model_class, model_args = model_args, 
            optimizer = optimizer, optimizer_args= optimizer_args, grad_scale=grad_scale, batch_size=batch_size, vbatch_size=vbatch_size,
            dampening_factor = dampening_factor,fold=fold, crystal_thresh=crystal_thresh, purge_distance=purge_distance, accum_neurons=accum_neurons,
            epochs=epochs, dataset_name=dataset_name, purge=purge, scale_grad=scale_grad, crystallize=crystallize, verbose=verbose, cuda_device=cuda_device)
    else:
        acc, res = run_sequencial(model_class =  model_class, model_args = model_args, 
           optimizer = optimizer, optimizer_args= optimizer_args, grad_scale=grad_scale, batch_size=batch_size, vbatch_size=vbatch_size,
           dampening_factor =dampening_factor,fold=fold, crystal_thresh=crystal_thresh, purge_distance=purge_distance, accum_neurons=accum_neurons,
           epochs=epochs, dataset_name=dataset_name, purge=purge, scale_grad=scale_grad, crystallize=crystallize, num_classes=num_classes, seq_num=seq_num, verbose=Falsverbosee, cuda_device=cuda_device)
        
    res['model_class'] = str_model_class
    res['optimizer'] = str_optimizer
    return acc, res


dataset_names= ['MNIST','CIFAR10', 'CIFAR100']
 
combinations = [
    [False, False], 
    [True, False],  
    [False, True], 
    [True, True], 
]
nets = ['AlexNetMini', 'resnet20', 'resnet32', 'resnet44', 'resnet56','FCNet']
accums = [0,2]    

kwargs = {
    "exp" : "whole",
    "model_class" : "resnet20", 
    "model_args" : {"in_feats":784, "shapes":[1000], "num_classes":10}, 
    "optimizer" : 'SGD', 
    "optimizer_args": {"lr":1e-3}, 
    "grad_scale":0.09, 
    "batch_size":100, 
    "vbatch_size":100,
    "dampening_factor" : 0.6,
    "fold":5, 
    "crystal_thresh":4.5e-5, 
    "purge_distance":14.0, 
    "accum_neurons":2,
    "epochs":200, 
    "dataset_name":"MNIST", 
    "purge":True, 
    "scale_grad":True, 
    "crystallize":False, 
    "verbose":False,
    "seq_num":2,
	"cuda_device":torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
}

def hms_from_seconds(time_el):
    h_el = int(time_el//3600)
    m_el = int((time_el-3600*h_el)//60)
    s_el = int(time_el-3600*h_el-m_el*60)
    return h_el, m_el, s_el

print(f"Running on {kwargs['cuda_device']}")

result_list = []
i=0
t0 = time()
num_runs = len(dataset_names) * len(nets) * len(accums) * len(combinations)
for net in nets:
    for dataset_name in dataset_names:
        for accum in accums:
            for combination in combinations:
                purge, scale_grad = combination
                t1 = time()
                time_el = int(t1-t0)
                h_el, m_el, s_el = hms_from_seconds(time_el)
                approx_time_to_go = 0 if i==0 else (time_el/i)*num_runs
                h_to, m_to, s_to = hms_from_seconds(approx_time_to_go)
                s = f"Working on Combination #{i:4d} of {num_runs:4d}: dataset {dataset_name} net: {net}, accum: {accum} purge: {purge}, scale_grad: {scale_grad} - time elapsed {h_el:2d}h {m_el:2d}min {s_el:2d}s - {h_to:2d}h {m_to:2d}min {s_to:2d} to go"
                print(s)
                logging.info(s)
                purge, scale_grad = combination
                kwargs['dataset_name']=dataset_name
                kwargs['model_class']=net
                kwargs['accum_neurons']=accum
                kwargs['purge']=purge
                kwargs['scale_grad']=scale_grad
                acc, result = run(**kwargs)
                result_list.append({'options':kwargs.copy(), 'acc':acc, 'result':result})
                pkl.dump(result_list, open('cv_res_e50_CIFAR100_tmp.pkl','wb'))
                i+=1
                print()
pkl.dump(result_list, open('cv_res.pkl','wb'))