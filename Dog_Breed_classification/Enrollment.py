#Title: Enrollment. Add new label to a trained model
#Date: 11/5/2022
#Author: Fischer @ Unico
#Usage: python3 Enrollment.py --new_samples [PATH] --database [PATH]

import argparse
import torch
from tqdm import tqdm
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from utils.utils import update_labels_file

def web_enrollment(path,database,labels):
    #check if there is more than one new label
    paths = path.split()
    for path in paths: web_enroll_new_label(path,database,labels)
#================================================
def web_enroll_new_label(path, database,labels):
    #get new label name from folder`s name
    new_label = os.path.basename(path)

    #check for existing label name
    if(new_label in os.listdir(database)):
        error = "Error: label name already exists"
        return error
    else:

        #create new label folder
        #new_label_path = os.path.join(args.database,args.new_label)
        new_label_train =os.path.join(database,'train',new_label)
        new_label_val = os.path.join(database,'val',new_label)
        os.mkdir(new_label_train)
        os.mkdir(new_label_val)

        #split the new samples in training and validation 
        new_samples_path = os.listdir(path)
        random.shuffle(new_samples_path)
        training_dataset ,test_dataset= train_test_split(new_samples_path,test_size=0.2)
        
        #copy files to database
        for a in training_dataset:
            new_train_path = os.path.join(new_label_train,a)
            old_train_path = os.path.join(path,a)
            shutil.copy(old_train_path,new_train_path)

        for b in test_dataset:     
            new_val_path = os.path.join(new_label_val,b)
            old_val_path = os.path.join(path,b)
            shutil.copy(old_val_path,new_val_path)
        
        #update labels file
        update_labels_file(database,labels)
        return "new label enrolled"

   # print("done")
   # print(f"new label: {new_label} was enrolled successfully")
   # print("The model must be retrained after all new labels are retrained")
#=============================================================
def main(): # DEBUGGING
    print("Start enrollment...")

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    #parser.add_argument("--new_label", type=str, required=True, help="New label string")
    parser.add_argument("--new_samples",  type=str, required=True, help=" dataset with pictures from new label")
    parser.add_argument("--database",  type=str, required=True, help=" database path")
    args=parser.parse_args()
    #-------------------------------------------

    #get new label name from folder`s name
    new_label = os.path.basename(args.new_samples)

    #check for existing label name
    if(new_label in os.listdir(args.database)):
        print("Error: label name already exists")
        return 
    else:

        #create new label folder
        #new_label_path = os.path.join(args.database,args.new_label)
        new_label_train =os.path.join(args.database,'train',new_label)
        new_label_val = os.path.join(args.database,'val',new_label)
        os.mkdir(new_label_train)
        os.mkdir(new_label_val)

        #split the new samples in training and validation 
        new_samples_path = os.listdir(args.new_samples)
        random.shuffle(new_samples_path)
        training_dataset ,test_dataset= train_test_split(new_samples_path,test_size=0.2)
        
        #copy files to database
        for a in training_dataset:
            new_train_path = os.path.join(new_label_train,a)
            old_train_path = os.path.join(args.new_samples,a)
            shutil.copy(old_train_path,new_train_path)

        for b in test_dataset:     
            new_val_path = os.path.join(new_label_val,b)
            old_val_path = os.path.join(args.new_samples,b)
            shutil.copy(old_val_path,new_val_path)
        
        #update labels file
        update_labels_file(args.database)

    print("done")
    print(f"new label: {new_label} was enrolled successfully")
    print("The model must be retrained after all new labels are retrained")
#====================================================
if __name__ == "__main__": main()