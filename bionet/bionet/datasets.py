import torch
from torchvision import datasets, transforms
from sklearn.model_selection import ShuffleSplit
class Datasets(torch.utils.data.Dataset):
    
    @staticmethod
    def filter_data(data, targets, cl_filter):
        if len(torch.tensor(cl_filter).shape) == 2:
            d_idxs = (targets.unsqueeze(0) == torch.tensor(cl_filter)[0].unsqueeze(1)).sum(0)==1
            t_idxs = (targets.unsqueeze(0) == torch.tensor(cl_filter)[1].unsqueeze(1)).sum(0)==1
            data = data[idxs]
            targets = targets[t_idxs]
        else:
            data = data[idxs]
            targets = targets[idxs]
        return data, targets    
    
    class MNIST():
        def __init__(self, path, download, cl_filter=None):
            super(Datasets.MNIST, self).__init__()
            self.trainset = datasets.MNIST(path, train=True, download=download, transform=transforms.ToTensor())
            self.valset = datasets.MNIST(path, train=False, download=download, transform=transforms.ToTensor())
            self.data = torch.cat((self.trainset.data,self.valset.data))/255
            self.targets = torch.cat((self.trainset.targets,self.valset.targets))
            self.train_idxs = torch.tensor(range(len(self.trainset)))
            self.val_idxs = torch.tensor(range(len(self.valset)))
            if cl_filter is not None:
                self.train_idxs = filter_data(self.train_idxs, self.trainset.targets[self.train_idxs], cl_filter)
                self.val_idxs = filter_data(self.val_idxs, self.valset.targets[self.val_idxs], cl_filter)
    
    class CIFAR10():
        def __init__(self,path,download, cl_filter=None):
            super(Datasets.CIFAR10, self).__init__()
            self.trainset = datasets.CIFAR10(path, train=True, download=download, transform=transforms.ToTensor())
            self.valset = datasets.CIFAR10(path, train=False, download=download, transform=transforms.ToTensor())
            self.data = torch.cat((torch.tensor(self.trainset.data),torch.tensor(self.valset.data))).movedim(-1,1)/255
            self.targets = torch.cat((torch.tensor(self.trainset.targets),torch.tensor(self.valset.targets)))
            self.train_idxs = torch.tensor(range(len(self.trainset)))
            self.val_idxs = torch.tensor(range(len(self.valset)))
            if cl_filter is not None:
                self.train_idxs = filter_data(self.train_idxs, self.trainset.targets[self.train_idxs], cl_filter)
                self.val_idxs = filter_data(self.val_idxs, self.valset.targets[self.val_idxs], cl_filter)
            
    class CIFAR100():
        def __init__(self,path,download, cl_filter=None):
            super(Datasets.CIFAR100, self).__init__()
            self.trainset = datasets.CIFAR100(path, train=True, download=download, transform=transforms.ToTensor())
            self.valset = datasets.CIFAR100(path, train=False, download=download, transform=transforms.ToTensor())            
            self.data = torch.cat((torch.tensor(self.trainset.data),torch.tensor(self.valset.data))).movedim(-1,1)/255
            self.targets = torch.cat((torch.tensor(self.trainset.targets),torch.tensor(self.valset.targets)))
            self.train_idxs = torch.tensor(range(len(self.trainset)))
            self.val_idxs = torch.tensor(range(len(self.valset)))
            if cl_filter is not None:
                self.train_idxs = filter_data(self.train_idxs, self.trainset.targets[self.train_idxs], cl_filter)
                self.val_idxs = filter_data(self.val_idxs, self.valset.targets[self.val_idxs], cl_filter)
            
            
    def set_normalization(self, name):
        if name == 'MNIST':
            self.normalization= {'mean':(0.1307,),'std':(0.3081,)}
        if name == 'CIFAR10':
            self.normalization= {'mean':(0.4914, 0.4822, 0.4465),'std':(0.2470, 0.2435, 0.2616)}   
        if name == 'CIFAR100':
            self.normalization= {'mean':(0.5071, 0.4865, 0.4409),'std':(0.2673, 0.2564, 0.2762)}   
        if name == 'Imagenet':
            self.normalization= {'mean':(0.485, 0.456, 0.406),'std':(0.229, 0.224, 0.225)}
            
    
    def initialize_dataset_based_on_name(self, name, path='../data', download=True,cl_filter=None):
        assert name.upper() in ['MNIST','CIFAR10', 'CIFAR100']
        dataset = getattr(Datasets, name.upper())(path, download)
        self.data=dataset.data
        self.targets=dataset.targets
        self.train_idxs=dataset.train_idxs
        self.val_idxs=dataset.val_idxs
        self.train_idxs_f=dataset.train_idxs
        self.val_idxs_f=dataset.val_idxs
        self.set_normalization(name)
    
    
    def __init__(self, dataset_name = 'MNIST', mode='train', path='../data', cv=0, download=True, cl_filter=None, verbose=False):
        super(Datasets, self).__init__()
        self.mode = 'train' if mode.lower()=='train' else 'test'
        if verbose == True: print(f'Inizializing Dataset in {mode} mode')        
        if cv>=2:
            self.init_cross_validation(n_fold=cv)                    

        self.data = None     
        self.targets= None
        self.train_idxs = None
        self.val_idxs = None
        self.train_idxs_f = None
        self.val_idxs_f = None
        self.transform = None
        self.train_mode = mode.lower()=='train'
        
        self.initialize_dataset_based_on_name(dataset_name,path=path, download=download, cl_filter=cl_filter)
        
    def init_cross_validation(self, n_fold):
        self.cross_val_iterator = self.get_cross_val_iterator(n_fold=n_fold)
        self.next_crossval_step()
        
    @staticmethod
    def filter_indices(idxs, targets, cl_filter:[]):
        d_idxs = (targets.unsqueeze(0) == cl_filter.unsqueeze(1)).sum(0)==1
        return idxs[d_idxs]
    
    def filter_dataset(self, cl_filter: [[],[]]):
        self.train_idxs_f = Datasets.filter_indices(self.train_idxs, self.targets[self.train_idxs], torch.tensor(cl_filter[0]))
        self.val_idxs_f = Datasets.filter_indices(self.val_idxs, self.targets[self.val_idxs], torch.tensor(cl_filter[1]))
        
        
    def get_cross_val_iterator(self, n_fold=5):
        return ShuffleSplit(n_fold,random_state=0).split(self.data, self.targets)
        
    def val(self):
        self.train_mode = False
        
    def train(self):
        self.train_mode = True
        
    def active_idxs(self):
        return self.train_idxs_f if self.train_mode else self.val_idxs_f
        
    def next_crossval_step(self):
        self.train_idxs, self.val_idxs = next(self.cross_val_iterator)
        self.train_idxs_f, self.val_idxs_f = self.train_idxs, self.val_idxs
        
    def __len__(self):
        return len(self.active_idxs())
    
    def __getitem__(self,idx):
        sample = self.data[self.active_idxs()[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[self.active_idxs()[idx]]
    
        