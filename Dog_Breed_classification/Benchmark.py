#Title: Benchmark 
#Date: 12/05/2022
#Author: Fischer @ Unico
#Usage: python3 Benchmark.py --mode [path] --dataset [PATH] --csv_out [NAME.csv] --model[PATH] --csv_path[PATH][OPTIONAL] --cfg[PATH][OPTIONAL] --classes [PATH][OPTIONAL] --cm [OPTIONAL]
#       python3 Benchmark.py --mode [csv] --dataset [PATH.csv] --cfg[PATH][OPTIONAL] --classes[PATH][OPTIONAL]

import argparse
import csv
import torch
import torchvision
from utils.inf_func import load_model
from utils.utils import to_device, read_cfg, get_model,get_classes
from Dataset_tools.preprocessing import test_data
from utils.DeviceDataLoader import DeviceDataLoader
from Model_arch.Tester import Tester
from utils.eval import metrics_csv
import os


def main():
    print("Starting Benckmark....")

    print("Pytorch Version:" + torch.__version__)
    print("Torchvision Version:" + torchvision.__version__)
    print("Cuda Version:" + torch.version.cuda)

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/dog_rec.yaml", help="Path to config(.yaml) file")
    parser.add_argument("--model",type=str, help="input trained model")
    parser.add_argument("--csv_path",type=str,default="benchmark_results/", help="csv path for benchmark results")
    parser.add_argument("--csv_out",type=str, help="output csv file name")
    parser.add_argument("--dataset", type=str,required = True, help="input benchmark dataset path to inference")
    parser.add_argument("--mode", choices=['path','csv'],required = True, help="input mode")
    parser.add_argument("--classes",type = str, default = "config/labels.txt", help="path from labels file")
    parser.add_argument("--cm", action = 'store_true', help= "confusion_matrix plot")
    args=parser.parse_args()
    #-------------------------------------------
    #checking args
    if(args.dataset is not None):
        print("Inferencing on dataset:" + args.dataset)
        dataset = args.dataset
    else :
        print("Error benckmark dataset was not inputed")
        print("Usage: python3 benckmark.py --dataset [PATH]")
        return

    if(args.mode == "path" and args.model is None): 
        print("Error no model available")
        return
    #-------------------------------------------
    print(f"input mode: {args.mode}")
    mode = args.mode

    if(mode =='path'):
        print(f"Reading config file {args.cfg} ....")
        cfg=read_cfg(cfg_file=args.cfg)
        print ("ok")

        #config GPU device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Using {} device'.format(device))

        #number of classes in the dataset
        #N_classes = get_classes(cfg)
        N_classes = len(os.listdir(args.dataset))
        print("Number of classes: " + str(N_classes))
        
        #load model architeture
        print("Loading model..." + args.model)
        network = get_model(cfg,N_classes)        
        network = load_model(network, args.model, device)
        print("Ok " + "model:" + cfg['model']['base'] + '\n')


        #load test data to GPU
        print("Load testing data...")
        testloader=test_data(dataset,cfg)
        test_dl = DeviceDataLoader(testloader, device)
        print(f"test size: {len(testloader)} batches")
        print("Ok")

        
        print("Testing...")
    
        tester = Tester ( cfg = cfg,
                        testloader= test_dl,
                        network = network,
                        device = device,
                        csv_path= args.csv_path,
                        csv_name = args.csv_out,
                        dataset = dataset,
                        labels = args.classes
                        )

        tester.test(cm_flag = args.cm)

    elif(mode == 'csv'): 
        print("Testing...")      

        #get classes file
        f = open(args.classes,'r')
        classes = f.readlines()
        for i in range(len(classes)): 
            tmp = classes[i]
            classes[i] = tmp[:-1]
        # perform metrics
        metrics_csv(dataset, classes)
    print("done")

#========================================================
if __name__=="__main__": main() 


