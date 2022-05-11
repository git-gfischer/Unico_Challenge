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

def main():
    print("Starting inference")

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="input image to inference")
    parser.add_argument("--model",type=str,default="experiments/exp_1/weights/resnet18_dogs_7.pth", help="input trained model")
    parser.add_argument("--network", choices=['resnet18','resnet50','effNet'], default='resnet18', help="network architecture")
    parser.add_argument("--classes", default='config/labels.txt', help="labels file")
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
    frame = cv2.imread(args.img)
    out, prob = prediction(network,frame,device)
    print(f"Result: {classes[out]} | prob: {prob.tolist()} ")

    #display image
    cv2.imshow('display',frame)
    cv2.waitKey(0)
    

#========================================================
if __name__=="__main__": main()


