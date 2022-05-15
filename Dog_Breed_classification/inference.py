#Title: Inference Dog Bread Recognition
#Date: 9/05/2022
#Author: Fischer @ Unico
#Usage: python3 inference.py --img[PATH] --model[PATH][OPTIONAL] --network[NAME][OPTIONAL]

import argparse
#from parso import parse
from Model_arch.resnet18_arch import Resnet18
from Model_arch.resnet50_arch import Resnet50
from Model_arch.Efficientnet_b2 import Efficientnet_b2
from utils.utils import to_device
from utils.inf_func import prediction,load_model,load_classes
import torch
import cv2
import time 

def web_inferece(img_bin,network,device,classes,th=0.4):
    # inferece function for web aplication

    #prediction
    start_time = time.time()
    out, prob = prediction(network,img_bin,device)
    prob_list = prob.tolist()[0]
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time} seconds")
    if(float(prob_list[out]) > th): 
        print(f"Result: {classes[out]} | prob: {prob_list} ")
        return classes[out], prob_list[out]
    else: 
        #print(f"class: {classes[out]}prob: {str(prob_list[out])}")
        print("Result: Unknown")
        return "Unknown",prob_list[out]
#======================================================================
def main():
    print("Starting inference")

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="input image to inference")
    parser.add_argument("--model",type=str,default="experiments/exp_1/weights/resnet18_dogs_7.pth", help="input trained model")
    parser.add_argument("--network", choices=['resnet18','resnet50','effNet'], default='resnet50', help="network architecture")
    parser.add_argument("--classes", default='config/labels.txt', help="labels file")
    parser.add_argument("--th",type=float,default= 0.4, help = "Threshold for model output ")
    args=parser.parse_args()
    #-------------------------------------------
    #checking args
    if(args.img is not None):
        print("Inferencing on an image:" + args.img)

    if(args.model is None): 
        print("Error no model available")
        return
    #--------------------------------------------
    print("Reading model: " + args.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))
    
    print("Loading model...")
    if args.network == 'resnet18':
        network = Resnet18(False)
    elif args.network == 'resnet50':
        network = Resnet50(False)
    elif args.network == 'effNet':
        network = Efficientnet_b2(False)

    #load network  
    network = to_device(network,device)
    network = load_model(network, args.model, device)

    print("Ok " + "model: " + args.network + '\n')

    #load labels
    classes = load_classes(args.classes) 

    #prediction
    start_time = time.time()
    frame = cv2.imread(args.img)
    out, prob = prediction(network,frame,device)
    prob_list = prob.tolist()[0]
    if(float(prob_list[out]) > args.th): print(f"Result: {classes[out]} | prob: {prob_list[out]} ")
    else: 
        #print(f"class: {classes[out]}prob: {str(prob_list[out])}")
        print("Result: Unknown")
        print(f"Debug: {classes[out]} Prob: {prob_list[out]}")
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time} seconds")

    #display image
    cv2.imshow('display',frame)
    cv2.waitKey(0)
    

#========================================================
if __name__=="__main__": main()


