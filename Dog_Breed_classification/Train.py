# Title: Dog Bread Classification Training Sketch
# Date: 9/05/2022
# Author : Fischer @ Unico
# Usage : python3 train.py --cfg[PATH][OPTIONAL] --logs_path[PATH][OPTINAL] --exp_name[NAME][OPTIONAL] --multiGPU [OPTIONAL]

#Todos:
# Retreinar Resnet18,Resnet50, effNet
# Web application 
# docker
# Readme
# YOLO  (Plus)
# API (Plus)

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import torchvision
from utils.utils import *
from Model_arch.Trainer import Trainer
from Dataset_tools.preprocessing import training_val_data 
from utils.DeviceDataLoader import DeviceDataLoader
from utils.Report_Class import Report_class

from torch.utils.tensorboard import SummaryWriter

def main():
    print("Starting Training")

    print("Pytorch Version:" + torch.__version__)
    print("Torchvision Version:" + torchvision.__version__)
    # print("Cuda Version:" + torch.version.cuda)

    #parsing arguments------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/dog_rec.yaml", help="Path to config(.yaml) file")
    parser.add_argument("--logs_path", type=str, default="experiments", help="Path where to save logs")
    parser.add_argument("--exp_name", type=str, default="exp_1", help="Name of experiment folder to save logs")
    parser.add_argument("--multiGPU", action = 'store_true', help= "enable multi GPU training")
    args = parser.parse_args()
    #-----------------------------------------------

    #Read config file
    print(f"Reading config file {args.cfg} ....")
    cfg=read_cfg(cfg_file=args.cfg)
    print ("Ok" + '\n')

    #config GPU device
    device = get_device(cfg)
    print('Using {} device'.format(device) + '\n')

    #number of classes in the dataset
    N_classes = get_classes(cfg)
    print("Number of classes: " + str(N_classes))

    #config network
    print("Load Network...")
    network=get_model(cfg,N_classes)
    print("Ok " + "model:" + cfg['model']['base'] + '\n')
   
    # config optimizer
    print("Load Optimizer...")
    optimizer=get_optimizer(cfg,network)
    print("Ok " + "optimizer:" + cfg['train']['optimizer']+ '\n')

    #config learning rate scheduler
    print("Scheduler Learning rate")
    lr_scheduler = get_scheduler(cfg,optimizer)
    print("Ok " + "scheduler: " + cfg['scheduler']['type']+ '\n')

    #config loss function
    print("Load Loss Function")
    loss_fn = get_loss_fn(cfg)
    print("Ok " + "loss function:" + cfg['train']['loss_fn']+ '\n')
    
    #Split data in training and validation
    print("Load training and Validation data...")
    trainloader,valloader=training_val_data(cfg)

    #load dataloader into GPU
    train_dl = DeviceDataLoader(trainloader, device)
    val_dl = DeviceDataLoader(valloader, device)
    print(f"Training size: {len(trainloader)} batches")
    print(f"Validation size: {len(valloader)} batches")
    print("Ok"+ '\n')


    logs_full_path = os.path.join(args.logs_path, args.exp_name)
    if not os.path.exists(logs_full_path): os.makedirs(logs_full_path)
    report_path = os.path.join(args.logs_path, args.exp_name, "report")
    if not os.path.exists(report_path): os.makedirs(report_path)

    print("Starting TensorBoard.....")
    writer = SummaryWriter(f'{logs_full_path}/Tensorboard_logs')
    print("Ok"+ f" Path: {logs_full_path}/Tensorboard_logs" +'\n')

    #training report 
    print("Creating training report.....")
    report = Report_class(cfg,report_path)
    print ("Ok")

    trainer= Trainer(cfg=cfg,
                    network=network,
                    optimizer=optimizer,
                    loss_fn =loss_fn,
                    device=device,
                    trainloader=train_dl,
                    valloader=val_dl,
                    lr_scheduler= lr_scheduler,
                    logs_path=logs_full_path,
                    multiGPU=args.multiGPU,
                    report = report,
                    writer=writer)

    print("Starting training...")
    trainer.train()
    print("Finish Training")
    writer.close()

    #save model
    #print("Saving model....")
    #trainer.save_model(cfg['train']['num_epochs'])
    #print("Model saved")

#==============================================
if __name__=='__main__': main()










