#Title: Benchmark with unlabeled data 
#Date: 16/05/2022
#Author: Fischer @ Unico
#Usage: python3 Benchmark_unlabeled.py --dataset [PATH] --unknown [PATH] --csv_out [NAME.csv] --model[PATH] --csv_path[PATH][OPTIONAL] --cfg[PATH][OPTIONAL] --classes [PATH][OPTIONAL]


import argparse
import csv
import torch
import torchvision
from utils.inf_func import load_model,load_classes
from utils.utils import to_device, read_cfg, get_model,get_classes
from Dataset_tools.preprocessing import test_data
from utils.DeviceDataLoader import DeviceDataLoader
from Model_arch.Tester import Tester
from utils.eval import metrics_csv
from inference import web_inference
from sklearn.metrics import classification_report
import cv2
import os


def main():
    print("Starting Benckmark....")

    print("Pytorch Version:" + torch.__version__)
    print("Torchvision Version:" + torchvision.__version__)
    print("Cuda Version:" + torch.version.cuda)

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/dog_rec.yaml", help="Path to config(.yaml) file")
    parser.add_argument("--th", type=float, default=0.5, help="Threshold for unknown data")
    parser.add_argument("--model",type=str, default = "experiments/resnet50_ext_SGD_001_LR5/weights/resnet50_dogs_16.pth",help="input trained model")
    parser.add_argument("--dataset", type=str,required = True, help="input benchmark dataset path to inference")
    parser.add_argument("--unknown", type=str,required = True, help="input unlabeled benchmark dataset path to inference")
    parser.add_argument("--classes",type = str, default = "config/labels.txt", help="path from labels file")

    args=parser.parse_args()
    #-------------------------------------------
    print(f"Reading config file {args.cfg} ....")
    cfg=read_cfg(cfg_file=args.cfg)
    print ("ok")

    #config GPU device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print('Using {} device'.format(device))

    #number of classes in the dataset
    #N_classes = get_classes(cfg)
    N_classes = len(os.listdir(args.dataset))
    classes = load_classes(args.classes)
    for i,clas in enumerate(classes) : classes[i] = clas.replace("\n","")
    print("Number of classes: " + str(N_classes))
    
    #load model architeture
    print("Loading model..." + args.model)
    network = get_model(cfg,N_classes)        
    network = load_model(network, args.model, device)
    print("Ok " + " model:" + cfg['model']['base'] + '\n')

    labels_vec = []
    scores = []
    print("Testing...")

    #inference on labeled data
    for (dirpath, dirnames, filenames) in os.walk(args.dataset):
        if(len(filenames) == 0): continue   # check if it is empty
        if(filenames[0].endswith(".jpg")==False): continue # check if image is not .jpg extension


        label = os.path.basename(os.path.normpath(dirpath))
        label_idx = classes.index(label)
        for filename in filenames: 
            labels_vec.append(label_idx)
            image = cv2.imread(os.path.join(dirpath,filename)) # read image
            pred,_=web_inference(image,network,device,classes,th=args.th) #inference
            if (pred == "Unknown"): score_idx = N_classes+1
            else: score_idx = classes.index(pred) 
            scores.append(score_idx)


    #inference on unlabeled data
    for (dirpath, dirnames, filenames) in os.walk(args.unknown):
        if(len(filenames) == 0): continue   # check if it is empty
        if(filenames[0].endswith(".jpg")==False): continue # check if image is not .jpg extension

        label_idx = N_classes+1        
        for filename in filenames:
            labels_vec.append(label_idx)
            image = cv2.imread(os.path.join(dirpath,filename)) # read image
            pred,_=web_inference(image,network,device,classes,th=args.th) #inference
            if (pred=="Unknown"): score_idx = N_classes+1
            else: score_idx = classes.index(pred) 
            scores.append(score_idx)

    classes.append("Unknown")
    print(classification_report(labels_vec, scores, target_names=classes))
    print("done")

#========================================================
if __name__=="__main__": main() 


