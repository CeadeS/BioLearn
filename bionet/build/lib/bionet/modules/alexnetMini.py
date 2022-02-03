import torch
from torch import nn
from torch.nn import functional as F
from bionet.biomodule import BioModule as bm

class AlexNetMini(nn.Module):
    def __init__(self, num_classes=10, num_chans=1, target='MNIST'):
        super(AlexNetMini, self).__init__()
        padding = 7 if target == 'MNIST' else 5
            
        self.conv1 = nn.Conv2d(num_chans, 96, kernel_size=11, stride=4, padding=padding)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        
        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv2,
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3,
            nn.ReLU6(inplace=True),
            self.conv4,
            nn.ReLU6(inplace=True),
            self.conv5,
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.lin1 = nn.Linear(384, 384)
        self.lin2 = nn.Linear(384, 384)
        self.lin3 = bm.mark_skip_accum(nn.Linear(384, num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.lin1(x)
        x = F.relu6(x)
        x = self.lin2(x)
        x = F.relu6(x)
        x = self.lin3(x)
        return x

        
    def __forward(self, x):
        x = self.conv1(x)
        x = F.relu6(x)
        x = F.max_pool2d(x,kernel_size = 2, stride = 2)
        x = self.conv2(x)
        x = F.relu6(x)
        x = F.max_pool2d(x,kernel_size = 2, stride = 2)
        x = self.conv3(x)
        x = F.relu6(x)
        x = self.conv4(x)
        x = F.relu6(x)
        x = self.conv5(x)
        x = F.relu6(x)
        x = F.max_pool2d(x,kernel_size = 2, stride = 2)
        x = torch.flatten(x,1)
        x = self.lin1(x)
        x = F.relu6(x)
        x = self.lin2(x)
        x = F.relu6(x)
        x = self.lin3(x)
        return x