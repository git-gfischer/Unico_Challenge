
from cProfile import label
from turtle import st

from sklearn import datasets
from Model_arch.base import BaseTrainer
#from utils.metrics import AvgMeter
from utils.utils import to_device
from utils.inf_func import predict_batch
from os import listdir,walk
from os.path import isfile, join
# from retinaface.retinaface import RetinaFace

#from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from tqdm import tqdm

import csv
import os
import torch 
import time


class Tester(BaseTrainer):
    def __init__(self, cfg, network, device, testloader, csv_path, csv_name, dataset,
                 model_path=None ,multiGPU=False):
        
        # Initialize variables
        self.cfg = cfg
        self.device = device
        self.testloader = testloader
        self.csv_path = csv_path
        self.csv_name = csv_name
        self.dataset = dataset
        
    
        if (multiGPU == True and torch.cuda.device_count() > 1):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = torch.nn.DataParallel(self.network)
        
        self.network = to_device(network,device)
#==============================================        
    def save_csv(self, filename, label, score, time):
        dataset_info = self.csv_name
        save_path = os.path.join(self.csv_path, dataset_info)
        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)
        
        header = ["filaname", "label","pred", 'inference_time_seconds']

        # open the file in the write mode
        with open(save_path, 'w') as f:
            # create the csv writer
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for i in range(len(filename)):
                # write a row to the csv file
                if(label[i]==0): r_label = "afro-american"
                elif(label[i]==1): r_label = "asian"
                elif(label[i]==2): r_label = "caucasian"
                elif(label[i]==3): r_label = "indian"
                if(score[i]==0): r_pred = "afro-american"
                if(score[i]==1): r_pred = "asian"
                if(score[i]==2): r_pred = "caucasian"
                if(score[i]==3): r_pred = "indian"

                tmp =[filename[i],r_label, r_pred, round(time[i],2)]
                writer.writerow(tmp)
        print("CSV saved")
#==============================================
    def test(self):
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
    
        #labels_list = np.concatenate(labels_list, axis=0)
        #scores = np.concatenate(scores, axis=0)

        print("Saving CSV...")
        self.save_csv(filenames_vec, labels_list, scores, time_vec)
    

        print(classification_report(labels_list, scores, target_names=["African", "Asian", "Caucasian", "Indian"]))
        print("Inference time: {:.2f} seconds \n".format(inference_mean/len(self.testloader)))