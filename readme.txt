### Installation
go to bionet/ and run pip install .

### Accuracy and learning progress
go to eval/ and run python3 experiment.py


### Gradient Reconstruction 
You need to save a trained model first.

train model with grad_rec/train_models_for_gradinversion.ipynb
Store this model state dict in grad_rec/models
run gradient inversion in SimpleReconstruction.ipynb


###

#Models available can be easily converted to bionet
#small example:

#/bin/python

import torch
from bionet.biomodule import BioModule
from torchvision.models.resnet import wide_resnet101_2, ResNet
from torchvision.models.mnasnet import mnasnet1_0, MNASNet

converter = BioModule.get_convert_to_bionet_converter(accum_neurons=2, out_feats=1000) ## do not use accumulation on layer with n output features

im = torch.rand((3,3,224,224))

converter(ResNet)
net = wide_resnet101_2()
net(im)

converter(MNASNet)
net_2 = mnasnet1_0()
net_2(im)
