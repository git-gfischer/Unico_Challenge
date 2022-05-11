import sys
from Model_arch.base import BaseTrainer
sys.path.append(".") # Adds higher directory to python modules path.
import csv
import os
import pdb
from random import randint
import time
import torch
from tqdm import tqdm
from utils.metrics import AvgMeter
from utils.utils import to_device, report_metrics
from utils.eval import calc_accuracy

from torch.utils.tensorboard import SummaryWriter


class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, loss_fn, device, trainloader, valloader,writer,lr_scheduler,
                 logs_path,report,model_path=None, multiGPU=False):
        super(Trainer, self).__init__(cfg, network, optimizer, loss_fn, device, trainloader,
                                      valloader,writer,lr_scheduler,report)
       

        if  multiGPU==True and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = torch.nn.DataParallel(self.network)
        
        self.network = to_device(network,device)
        
        self.writer = writer

        # ! AvgMeter not using tensorboard writer
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train',
                                          num_iter_per_epoch=len(self.trainloader),
                                          per_iter_vis=True)

        self.train_acc_metric = AvgMeter(writer=writer, 
                                         name='Accuracy/train', num_iter_per_epoch=len(self.trainloader),
                                         per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))
                
        self.logs_path = logs_path

        self.weights_path = os.path.join(logs_path, 'weights')
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
            
        if model_path is not None:
            self.load_model(model_path)
            print(f"Model {model_path} loaded!")
        else:
            self.last_epoch = 0
#-----------------------------------------------------------       
    def load_model(self, model_path):
        state = torch.load(model_path)

        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])
        if(self.scheduler is not None): self.lr_scheduler.load_state_dict(state['scheduler'])
        self.last_epoch = state['epoch'] + 1
#----------------------------------------------------------- 
    def save_model(self, epoch):
        saved_name = os.path.join(self.weights_path,
                                  f"{self.cfg['model']['base']}_"\
                                  f"{self.cfg['dataset']['name']}_{epoch}.pth")
        
        torch.save(self.network.state_dict(),saved_name)       
#-----------------------------------------------------------     
    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        progress_bar = tqdm(total = len(self.trainloader))
        for batch in self.trainloader:
            loss,accuracy = self.network.training_step(batch,self.loss_fn)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            progress_bar.set_description(f"loss {self.train_loss_metric.avg:.4f} acc: {self.train_acc_metric.avg:.4f}")
            progress_bar.update()

            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy.item())

        epoch_loss = self.train_loss_metric.avg
        epoch_acc = self.train_acc_metric.avg
        return epoch_loss , epoch_acc
#-----------------------------------------------------------     
    def validate_one_epoch(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)
        
        with torch.no_grad():
            outputs = [self.network.validation_step(batch, self.loss_fn) for batch in self.valloader]
            result = self.network.validation_epoch_end(outputs)
            
            self.val_loss_metric.update(result['val_loss'])
            self.val_acc_metric.update(result['val_acc'])

            return self.val_loss_metric.avg, self.val_acc_metric.avg 
#-----------------------------------------------------------         
    def train(self):
        
        min_loss = float('inf')
 
        for epoch in range(self.last_epoch, self.cfg['train']['num_epochs']):

            train_epoch_loss, train_epoch_acc = self.train_one_epoch(epoch)

            val_epoch_loss, val_epoch_acc = self.validate_one_epoch(epoch)

            print(f'\nEpoch: {epoch}, Train loss: {train_epoch_loss:.4f}, Val loss: {val_epoch_loss:.4f}, '\
                f'Train Acc: {train_epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f} \n')
            
            if(self.lr_scheduler is not None): 
                self.lr_scheduler.step()
                lr = self.lr_scheduler.get_last_lr()
                print(f'lr: {lr}\n')
            else: lr = self.cfg['train']['lr']
                
            self.report.report_metrics(epoch, train_epoch_loss, val_epoch_loss,train_epoch_acc, val_epoch_acc, lr)

            self.writer.add_scalar('Loss/Train', train_epoch_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_epoch_loss, epoch)
            
            self.writer.add_scalar('Acc/Train', train_epoch_acc, epoch)
            self.writer.add_scalar('Acc/Val', val_epoch_acc, epoch)

           
            if val_epoch_loss < min_loss: # save models with lowest validation loss
                min_loss = val_epoch_loss
                self.save_model(epoch)
                print(f"Saved model: {epoch} \n\n")
    

