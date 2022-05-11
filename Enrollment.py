#Title: Enrollment. Add new label to a trained model
#Date: 11/5/2022
#Author: Fischer @ Unico
#Usage: python3 Enrollment.py --new_label [STR] --dataset [PATH] --model [PATH][OPTIONAL]

import argparse
import torch
from tqdm import tqdm

def main():
    print("Start enrollment...")

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--new_label", type=str, required=True, help="New label string")
    parser.add_argument("--model",type=str,default="experiments/exp_1/weights/resnet18_dogs_7.pth", help="input trained model")
    parser.add_argument("--dataset",  type=str, required=True, help=" dataset with pictures from new label")
    args=parser.parse_args()
    #-------------------------------------------

    #load the model


    print("done")
#====================================================
if __name__ == "__main__": main()