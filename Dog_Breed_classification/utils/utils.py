from xml.sax.xmlreader import IncrementalParser
import yaml
import torch
from torch import nn
from torch import optim
import numpy as np
from Model_arch.resnet18_arch import Resnet18
from Model_arch.resnet50_arch import Resnet50
from Model_arch.Efficientnet_b2 import Efficientnet_b2
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import cv2
import matplotlib.pyplot as plt
from skimage import transform as trans
import time
import os

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg
#===================================================== 
def save_cfg(cfg_dict, save_path):
    '''
        Save cfg in yaml file
    '''
    with open(save_path, 'w') as yaml_file:
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

#====================================================
def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(network.parameters(), lr=cfg['train']['lr'], momentum = 0.9, weight_decay=0.9)
    elif cfg['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(network.parameters(),lr=cfg['train']['lr'], momentum = 0.9)
    else:
        raise NotImplementedError

    return optimizer
#=====================================================
def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    device = None
    if len(cfg['device']) <= 1:
        if cfg['device'] == []:
            device = torch.device("cpu")
        elif cfg['device'][0] == 0:
            device = torch.device("cuda:0")
        elif cfg['device'][0] == 1:
            device = torch.device("cuda:1")
    elif len(cfg['device']) > 1:
        # device = [torch.device(f"cuda:{cfg['device'][0]}"), torch.device(f"cuda:{cfg['device'][1]}")]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise NotImplementedError
    # print(f"GPUs found: {device}")
    return device
#====================================================
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
#===================================================
def get_loss_fn(cfg):
    """Get Loss Function"""
    loss_fn = None
    if cfg['train']['loss_fn'] == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        raise NotImplementedError
    return loss_fn  
#===================================================
def get_model(cfg,N_classes):
    """Get Model architeture"""
    network = None
    if cfg['model']['base'] == 'resnet18':
        network = Resnet18(cfg['model']['pretrained'],n_classes = N_classes)
    elif(cfg['model']['base']=='resnet50'):
        network = Resnet50(cfg['model']['pretrained'],n_classes = N_classes)
    elif(cfg['model']['base']=='effNet'):
        network = Efficientnet_b2(cfg['model']['pretrained'],n_classes = N_classes)
    else:
        raise NotImplementedError
    return network
#===================================================
def get_scheduler(cfg, optimizer):
    """Get learning rate scheduler"""
    scheduler = None
    if cfg['scheduler']['type'] == 'None':
        return None
    elif cfg['scheduler']['type'] == 'linear':
        scheduler = StepLR(optimizer,step_size = cfg['scheduler']['step'], gamma = cfg['scheduler']['decay'])
    elif cfg['scheduler']['type'] == 'exponential':
        scheduler = ExponentialLR(optimizer,gamma= cfg['scheduler']['decay'])
    else:
        raise NotImplementedError
    return scheduler
#===================================================
def get_classes(cfg):
    """Get number of classes in the dataset"""
    dataset_folder = cfg['dataset']['root']
    dataset_folder = os.path.join(dataset_folder,'train')
    return len(os.listdir(dataset_folder))
#===================================================
def update_labels_file(dataset_folder, labels_path = "config/labels.txt"):
    """Update labels file with new labels"""
    dataset_folder = os.path.join(dataset_folder,'train')
    labels_names = sorted(os.listdir(dataset_folder))

    f = open(labels_path,'w')
    for label in labels_names: f.write(label + '\n')
#======================= Plot ========================
#=====================================================
def plot_chart(train, val, num_epochs, metric="Loss",plot=True):
    epoch = range(num_epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(val, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.xticks(epoch)
    plt.xlim(0, num_epochs-1)
    plt.grid()
    plt.legend()
    if plot: plt.show()
#======================= Report ======================
#=====================================================
def report_header(cfg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    path = os.path.join(cfg['report']['path'],cfg['report']['name'])
    file = open(path, 'a')

    # Timestamp // Path Dataset // Model // Batch Size // Optimizer // Loss Funciton // Scheduler // Learning Rate
    file.write('Timestamp: ' + timestamp + '\n')
    file.write('Path Dataset: ' + cfg['dataset']['root'] + '\n')
    file.write('Model: ' + cfg['model']['base'] + '\n')
    file.write('Batch Size: ' + str(cfg['test']['batch_size']) + '\n')
    file.write('Optimizer: ' + cfg['train']['optimizer'] + '\n')
    file.write('Loss Function: ' + cfg['train']['loss_fn'] + '\n')
    file.write('Scheduler[type/step/decay]: ' + cfg['scheduler']['type'] + '/' + str(cfg['scheduler']['step']) + '/' + str(cfg['scheduler']['decay']) + '\n')
    file.write('Learning Rate: ' + str(cfg['train']['lr']) + '\n')
    file.write('Epochs: ' + str(cfg['train']['num_epochs']) + '\n')
    file.close()

def report_metrics(cfg, epoch, train_loss, val_loss, lr = ''):
    path = os.path.join(cfg['report']['path'],cfg['report']['name'])
    file = open(path, 'a')
    file.write('Epoch: ' + str(epoch) + '    Train Loss: ' + str(train_loss) + '  Val Loss: ' + str(val_loss) + str(lr) + '\n')
    file.close()