import os
import torchvision
from torch.utils.data import DataLoader , ConcatDataset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from Dataset_tools.DataLoader import ImageFolderWithPaths

def training_val_data(cfg):
    train_transform = Compose ([Resize(cfg['model']['input_size']),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
    
    val_transform = Compose([Resize(cfg['model']['input_size']),
                            ToTensor(),
                            Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])]) 
    
    data_dir = cfg['dataset']['root']

    if(cfg['dataset']['split'] == False):
        train_datasets = ImageFolderWithPaths(os.path.join(data_dir, 'train'), train_transform)
        val_datasets =  ImageFolderWithPaths(os.path.join(data_dir, 'test'), val_transform)
    else: 
        datasets = ImageFolderWithPaths(data_dir, val_transform)
        ds = train_val_dataset(datasets)
        train_datasets = ds['train']
        val_datasets =  ds['val']

    trainloader = DataLoader(dataset=train_datasets,
                                    batch_size=cfg['train']['batch_size'],
                                    shuffle=True,
                                    num_workers=cfg['dataset']['num_workers'],
                                    pin_memory=True)

    valloader = DataLoader(dataset=val_datasets,
                                    batch_size=cfg['val']['batch_size'],
                                    shuffle=True,
                                    num_workers=cfg['dataset']['num_workers'],
                                    pin_memory=True)



    return trainloader,valloader
#======================================================================
def test_multidataset_data(path,cfg):
    test_transform = Compose ([Resize(cfg['model']['input_size']),
                               ToTensor(),
                               Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
    
    l =[]
    test_datasets1 =  ImageFolderWithPaths(os.path.join(path, 'Validation'), test_transform)
    test_datasets2 =  ImageFolderWithPaths(os.path.join(path, 'Training'),   test_transform)
    l.append(test_datasets1)
    l.append(test_datasets2)
    image_dataset  = ConcatDataset(l)

    
    testloader = DataLoader(dataset=image_dataset,
                            batch_size=cfg['test']['batch_size'],
                            num_workers=2)
    return testloader
#======================================================================
def test_data(path,cfg):
    test_transform = Compose([Resize(cfg['model']['input_size']),
                               ToTensor(),
                               Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])

    test_datasets =  ImageFolderWithPaths(path, test_transform)

    testloader = DataLoader(dataset=test_datasets,
                            batch_size=cfg['test']['batch_size'],
                            num_workers=2)

    return testloader
#======================================================================
def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

    

