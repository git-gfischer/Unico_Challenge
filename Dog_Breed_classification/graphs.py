#Title: graphs.py / generate Loss and Accs graphs from report file
#Date: 9/05/2022
#Author: Fischer @ Unico 
#Usage: python3 graphs.py --report [PATH]

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils import plot_chart

def main():
    print("Generating Loss and Acc graphs for MTL model....")

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True, help="input report file path")
    args=parser.parse_args()
    #--------------------------------------------
    print("Reading report file: " + args.report)

    try: file_in = pd.read_csv(args.report,skiprows=9)
    except: 
        print("Error: Could not open report file")
        return 
    
    Epoch = file_in['Epoch'].tolist()
    max_Epoch = max(Epoch)

    Train_Loss = file_in['Train_Loss'].tolist()
    Val_Loss = file_in['Val_Loss'].tolist()
    Train_acc = file_in['Train_acc'].tolist()
    Val_acc = file_in['Val_acc'].tolist()

    plot_chart(Train_Loss,Val_Loss,max_Epoch, plot=False)
    plot_chart(Train_acc,Val_acc,max_Epoch, metric="Acc")

    print("done")
    
#==============================================================
if __name__=="__main__": main() 