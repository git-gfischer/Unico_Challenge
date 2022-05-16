
from cProfile import label
from turtle import st

from sklearn import datasets
from Model_arch.base import BaseTrainer
#from utils.metrics import AvgMeter
from utils.utils import to_device
from utils.inf_func import predict_batch
from os import listdir,walk
from os.path import isfile, join

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn


from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import os
import torch 
import time
import itertools

class Tester(BaseTrainer):
    def __init__(self, cfg, network, device, testloader, csv_path, csv_name, dataset,labels,
                 model_path=None ,multiGPU=False):
        
        # Initialize variables
        self.cfg = cfg
        self.device = device
        self.testloader = testloader
        self.csv_path = csv_path
        self.csv_name = csv_name
        self.dataset = dataset
        
        #labels file
        f = open(labels,'r')
        self.labels = f.readlines()
        for i in range(len(self.labels)): 
            tmp = self.labels[i]
            self.labels[i] = tmp[:-1]
        
        #multi GPU 
        if (multiGPU == True and torch.cuda.device_count() > 1):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = torch.nn.DataParallel(self.network)
        
        self.network = to_device(network,device)
#==============================================        
    def save_csv(self, filename, label, score, time):
        save_path = os.path.join(self.csv_path, self.csv_name)
        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)
        
        header = ["filename", "label","pred", 'inference_time_seconds']

        # open the file in the write mode
        with open(save_path, 'w') as f:
            # create the csv writer
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for i in range(len(filename)):
                # write a row to the csv file
                r_label = self.labels[label[i]]
                r_pred = self.labels[score[i]]

                tmp =[filename[i],r_label, r_pred, round(time[i],4)]
                writer.writerow(tmp)
        print("CSV saved")
#==============================================
    def plot_confusion_matrix(self,labels_list,scores):
        print("Showing confusion matrix")
        cf_matrix = confusion_matrix(torch.as_tensor(labels_list), torch.as_tensor(scores))
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in self.labels],
                    columns = [i for i in self.labels])
        plt.figure(figsize = (12,7))

        sn.heatmap(df_cm, annot=True)
        png_name = self.csv_name.replace(".csv",".png")
        save_path = os.path.join(self.csv_path, png_name)
        plt.savefig(save_path)
#==============================================
    def save_confusion_matrix(self,labels_list,scores):
        print("Saving confusion matrix")
        cf_matrix = confusion_matrix(torch.as_tensor(labels_list), torch.as_tensor(scores))
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in self.labels],
                    columns = [i for i in self.labels])

        pkl_name = self.csv_name.replace(".csv",".pkl")
        save_path = os.path.join(self.csv_path, pkl_name)
        df_cm.to_pickle(save_path) #save confusion matrix in pickel format
        print(f"confusion matrix: {save_path} saved")
#==============================================
    def test(self,cm_flag=False):
        filenames_vec = []
        labels_list = []
        scores = []
        time_vec = []
        inference_mean = 0

        pbar = tqdm(total=len(self.testloader))
        for i, (batch) in enumerate(self.testloader):
            #print(f"{i+1} / {len(self.testloader)}")
            start = time.time()
            images,labels = batch

            gt, _ = self.testloader.dl.dataset.samples[i]
            filename = os.path.basename(gt)
            filenames_vec.append(filename)

            preds = predict_batch(images,self.network)
            end = time.time()

            labels_list.append(labels.item())
            scores.append(preds.item())
            inference_time = end - start
            time_vec.append(inference_time)

            inference_mean += inference_time
            pbar.update()

        pbar.close()
    
        print("Saving CSV...")
        self.save_csv(filenames_vec, labels_list, scores, time_vec)

        if(cm_flag): # confusion matrix flag
            self.save_confusion_matrix(labels_list,scores)
            self.plot_confusion_matrix(labels_list,scores)
            
    
        print(classification_report(labels_list, scores, target_names=self.labels))
        print("Inference time: {:.4f} seconds \n".format(inference_mean/len(self.testloader)))