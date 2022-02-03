import torch
from .biomodule import BioModule
from .modules.linear import FCNet
from .datasets import Datasets
from torch import nn
from torchvision import transforms
import numpy as np

def val(model, dataloader, transform, verbose=False):  
    device = next(model.parameters())[0].device    
    with torch.no_grad():
        correct = 0
        loss = 0
        for idx, (sample, target) in enumerate(dataloader):
            sample, target= transform(sample.to(device)), target.to(device)
            if len(sample.shape) == 3:
                sample = sample.unsqueeze(1)
            output = model(sample)            
            loss += nn.CrossEntropyLoss()(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            correct_batch = pred.eq(target.view_as(pred)).sum().item()
            correct+=correct_batch
            if verbose: print("Val Targets:",set(list(target.cpu().tolist())))
        if verbose: print(f'Validation Accuracy: {correct/dataloader.batch_size/len(dataloader): 6.4f} Loss: {loss/dataloader.batch_size/len(dataloader):4.4f}')
        return correct/dataloader.batch_size/len(dataloader), loss.item()/dataloader.batch_size/len(dataloader)
    
def train(model, dataloader, transform, epochs=2, optimizer=torch.optim.Adam, optimizer_args={'lr':5e-4}, purge=True, scale_grad=True, crystallize=True, verbose=False):
    optimizer=optimizer(model.parameters(),**optimizer_args)
    device = next(model.parameters())[0].device    
    epoch_losses = []
    for epoch in range(epochs):
        correct=0
        step_losses=[]
        for idx, (sample, target) in enumerate(dataloader):
            model.zero_grad()
            sample, target= transform(sample.to(device)), target.to(device)
            if len(sample.shape) == 3:                
                sample = sample.unsqueeze(1)
            output = model(sample)            
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            if scale_grad: model.scale_grad()
            if purge: model.purge()
            if crystallize: model.crystallize()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True) 
            correct_batch = pred.eq(target.view_as(pred)).sum().item()
            correct+=correct_batch
            step_losses.append(loss.item())
        if verbose: print(f'Accuracy for Epoch {epoch:3d} : {correct/dataloader.batch_size/len(dataloader): 6.4f} ')
        epoch_losses.append(step_losses)
    return epoch_losses
            
def cross_validate_sequencial(model_class, model_args, dataset, transform, num_classes=10, seq_num=2,fold=5,  optimizer=torch.optim.SGD, optimizer_args={'lr':1e-3}, 
                   batch_size=100, vbatch_size=1000, epochs=2, purge=True, scale_grad=True, crystallize=True, verbose=False, cuda_device=None):
    dataset.init_cross_validation(fold)    
    fold_results=[]
    for fold in range(fold):
        seq_results = []
        for seq in range((num_classes)//seq_num):
            train_filt = list(range(seq*2,seq*2+seq_num))
            val_filt = list(range(seq*2+seq_num))            
            model = model_class(**model_args).cuda(cuda_device) if cuda_device != 'cpu' else model_class(**model_args).cpu()
            dataset.filter_dataset([train_filt,val_filt])
            dataset.train()
            dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=0, shuffle=False, batch_size=batch_size)
            train(model, dataloader, transform, epochs=epochs, optimizer=optimizer, optimizer_args=optimizer_args, 
                  purge=purge, scale_grad=scale_grad, crystallize=crystallize, verbose=verbose)
            dataset.val()
            dataloader = torch.utils.data.DataLoader(dataset,  pin_memory=True, num_workers=0, shuffle=True, batch_size=vbatch_size)
            seq_results.append(val(model, dataloader, transform,verbose=verbose))
            del model
        fold_results.append(seq_results)
        try:
            dataset.next_crossval_step()
        except:
            break
    results = torch.tensor(fold_results)
    return results[:,:,0],results[:,:,1]

def cross_validate_continuous(model_class, model_args ,dataset, transform,fold=5,  optimizer=torch.optim.SGD, optimizer_args={'lr':1e-3}, 
                   batch_size=100, vbatch_size=1000, epochs=200, purge=True, scale_grad=True, crystallize=True, verbose=False,cuda_device=None):
    dataset.init_cross_validation(fold)    
    results = []
    for fold_idx in range(fold):
        model = model_class(**model_args).cuda(cuda_device) if cuda_device != 'cpu' else model_class(**model_args).cpu()
        fold_results = []
        for e in range(epochs):
            dataset.train()
            dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=0, shuffle=True, batch_size=batch_size)
            epoch_losses = train(model, dataloader, transform, epochs=1, optimizer=optimizer, optimizer_args=optimizer_args, 
                  purge=purge, scale_grad=scale_grad, crystallize=crystallize, verbose=verbose)
            dataset.val()
            dataloader = torch.utils.data.DataLoader(dataset,  pin_memory=True, num_workers=0, shuffle=False, batch_size=vbatch_size)
            fold_results.append({"fold_idx":fold_idx, "val_loss":val(model, dataloader, transform,verbose=verbose), "train_loss":epoch_losses,"epoch":e})
        del model
        results.append(fold_results)
        try:
            dataset.next_crossval_step()
        except:
            break     
        
    return results

def cross_validate(model_class, model_args ,dataset, transform,fold=5,  optimizer=torch.optim.SGD, optimizer_args={'lr':1e-3}, 
                   batch_size=100, vbatch_size=1000, epochs=200, purge=True, scale_grad=True, crystallize=True, verbose=False,cuda_device=None):
    dataset.init_cross_validation(fold)    
    results = []
    for fold in range(fold):
        model = model_class(**model_args).cuda(cuda_device) if cuda_device != 'cpu' else model_class(**model_args).cpu()
        dataset.train()
        dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=0, shuffle=True, batch_size=batch_size)
        train(model, dataloader, transform, epochs=epochs, optimizer=optimizer, optimizer_args=optimizer_args, 
              purge=purge, scale_grad=scale_grad, crystallize=crystallize, verbose=verbose)
        dataset.val()
        dataloader = torch.utils.data.DataLoader(dataset,  pin_memory=True, num_workers=0, shuffle=False, batch_size=vbatch_size)
        results.append(val(model, dataloader, transform,verbose=verbose))
        del model
        try:
            dataset.next_crossval_step()
        except:
            break
    return zip(*results)

def run_sequencial(model_class =  FCNet, model_args = {'in_feats':784, 'shapes':[1000], 'num_classes':10}, 
        optimizer = torch.optim.SGD, optimizer_args= {'lr':1e-3}, grad_scale=0.09, batch_size=100, vbatch_size=1000,
        dampening_factor = 0.6,fold=5, crystal_thresh=4.5e-5, purge_distance=8.0, accum_neurons=2,
       epochs=2, dataset_name='MNIST', purge=True, scale_grad=True, crystallize=True, num_classes=10, seq_num=2, 
                   verbose=False, cuda_device=None):
    converter = BioModule.get_convert_to_bionet_converter(grad_scale=grad_scale, dampening_factor=dampening_factor, crystal_thresh=crystal_thresh,purge_distance=purge_distance, accum_neurons=accum_neurons)
    bioFCNet = converter(model_class)
    dataset = Datasets(dataset_name)
    transform=nn.Sequential(
            transforms.Normalize(**dataset.normalization)
            )
    accs, losses = cross_validate_sequencial(model_class=bioFCNet, model_args=model_args, 
                                  dataset=dataset, transform=transform, num_classes=num_classes, seq_num=seq_num,
                                  fold=fold,  optimizer=optimizer, optimizer_args=optimizer_args, 
                                  batch_size=batch_size, vbatch_size=vbatch_size, epochs=epochs, 
                                  purge=purge, scale_grad=scale_grad, crystallize=crystallize, verbose=verbose,cuda_device=cuda_device)
    options = {'model_class' :  model_class, 'model_args' : model_args, 'optimizer' :optimizer, 'optimizer_args': optimizer_args, 'grad_scale':grad_scale, 
           'batch_size':batch_size, 'vbatch_size':vbatch_size,'dampening_factor':dampening_factor,'fold':fold, 'crystal_thresh':crystal_thresh, 'purge_distance':purge_distance, 'accum_neurons':accum_neurons,'epochs':epochs, 
           'dataset_name':dataset_name, 'purge':purge, 'scale_grad':scale_grad, 'crystallize':crystallize, 'verbose':verbose}
    if verbose:
        print(np.mean(accs), np.std(accs))
        plt.plot(accs)
        plt.show()
        plt.plot(losses)
        plt.show()
    return accs[:,-1].mean(), {'accs':accs, 'losses':losses, 'options':options }

def run_whole(model_class =  FCNet, model_args = {'in_feats':784, 'shapes':[1000], 'num_classes':10}, 
        optimizer = torch.optim.SGD, optimizer_args= {'lr':1e-3}, grad_scale=0.09, batch_size=100, vbatch_size=1000,
        dampening_factor = 0.6,fold=5, crystal_thresh=4.5e-5, purge_distance=8.0, accum_neurons=2,
       epochs=50, dataset_name='MNIST', purge=True, scale_grad=True, crystallize=True, verbose=False, cuda_device=None):
    converter = BioModule.get_convert_to_bionet_converter(grad_scale=grad_scale, dampening_factor=dampening_factor, crystal_thresh=crystal_thresh,purge_distance=purge_distance, accum_neurons=accum_neurons)
    bioFCNet = converter(model_class)
    dataset = Datasets(dataset_name)
    transform=nn.Sequential(
            transforms.Normalize(**dataset.normalization)
            )
    accs, losses = cross_validate(model_class=bioFCNet, model_args=model_args ,dataset=dataset, transform=transform,
                                  fold=fold,  optimizer=optimizer, optimizer_args=optimizer_args, 
                                  batch_size=batch_size, vbatch_size=vbatch_size, epochs=epochs, 
                                  purge=purge, scale_grad=scale_grad, crystallize=crystallize, verbose=verbose,cuda_device=cuda_device)
    options = {'model_class' :  model_class, 'model_args' : model_args, 'optimizer' :optimizer, 'optimizer_args': optimizer_args, 'grad_scale':grad_scale, 
           'batch_size':batch_size, 'vbatch_size':vbatch_size,'dampening_factor':dampening_factor,'fold':fold, 'crystal_thresh':crystal_thresh, 'purge_distance':purge_distance, 'accum_neurons':accum_neurons,'epochs':epochs, 
           'dataset_name':dataset_name, 'purge':purge, 'scale_grad':scale_grad, 'crystallize':crystallize, 'verbose':verbose}
    if verbose:
        print(np.mean(accs), np.std(accs))
        plt.plot(accs)
        plt.show()
        plt.plot(losses)
        plt.show()
    return np.mean(accs), {'accs':accs, 'losses':losses, 'options':options }

def run_whole_continuous(model_class =  FCNet, model_args = {'in_feats':784, 'shapes':[1000], 'num_classes':10}, 
        optimizer = torch.optim.SGD, optimizer_args= {'lr':1e-3}, grad_scale=0.09, batch_size=100, vbatch_size=1000,
        dampening_factor = 0.6,fold=5, crystal_thresh=4.5e-5, purge_distance=8.0, accum_neurons=2,
       epochs=50, dataset_name='MNIST', purge=True, scale_grad=True, crystallize=True, verbose=False, cuda_device=None):
    converter = BioModule.get_convert_to_bionet_converter(grad_scale=grad_scale, dampening_factor=dampening_factor, crystal_thresh=crystal_thresh,purge_distance=purge_distance, accum_neurons=accum_neurons)
    bioFCNet = converter(model_class)
    dataset = Datasets(dataset_name)
    transform=nn.Sequential(
            transforms.Normalize(**dataset.normalization)
            )
    results = cross_validate_continuous(model_class=bioFCNet, model_args=model_args ,dataset=dataset, transform=transform,
                                  fold=fold,  optimizer=optimizer, optimizer_args=optimizer_args, 
                                  batch_size=batch_size, vbatch_size=vbatch_size, epochs=epochs, 
                                  purge=purge, scale_grad=scale_grad, crystallize=crystallize, verbose=verbose, cuda_device=cuda_device)
    options = {'model_class' :  model_class, 'model_args' : model_args, 'optimizer' :optimizer, 'optimizer_args': optimizer_args, 'grad_scale':grad_scale, 
           'batch_size':batch_size, 'vbatch_size':vbatch_size,'dampening_factor':dampening_factor,'fold':fold, 'crystal_thresh':crystal_thresh, 'purge_distance':purge_distance, 'accum_neurons':accum_neurons,'epochs':epochs, 
           'dataset_name':dataset_name, 'purge':purge, 'scale_grad':scale_grad, 'crystallize':crystallize, 'verbose':verbose}
    return {'results':results, 'options':options }