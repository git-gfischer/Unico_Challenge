#Title: Organize dataset to pytorch training
#Date: 9/5/2022
#Author: Fischer @ Unico
#Usage: python3 Organize.py --dataset [PATH] --black_img_filter [OPTIONAL]

import os 
import shutil
from tqdm import tqdm
import argparse
import cv2
import random
from sklearn.model_selection import train_test_split

def black_img_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(image) == 0:  return True
    else: return False
#=======================================   
def count_files(dir_path):
    count = 0
    for root_dir, cur_dir, files in os.walk(dir_path): count += len(files)
    return count
#=======================================
def main():
    print("Start organizing dataset ")

    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="input dataset path")
    parser.add_argument("--black_img_filter", action = 'store_true', help= "black image filter")
    args=parser.parse_args()
    #---------------------------------

    #create organized dataset folder
    org_dataset = os.path.basename(os.path.normpath(args.dataset))
    dataset_path = os.path.dirname(args.dataset)
    org_dataset = org_dataset + "_organized"
    org_dataset_path  = os.path.join(dataset_path,org_dataset)   
    if(os.path.exists(org_dataset_path)==False): os.mkdir(org_dataset_path)

    #crete training folder 
    train_dataset_path = os.path.join(org_dataset_path,"train")
    if(os.path.exists(train_dataset_path)==False): os.mkdir(train_dataset_path)

    #create validation folder
    val_dataset_path = os.path.join(org_dataset_path,"val")
    if(os.path.exists(val_dataset_path)==False): os.mkdir(val_dataset_path)

    #create test folder
    test_dataset_path = os.path.join(org_dataset_path,"test")
    if(os.path.exists(test_dataset_path)==False): os.mkdir(test_dataset_path)

    n_files = count_files(args.dataset)
    print(f"number of files: {n_files}")

    # go through every image in the original dataset
    progress_bar = tqdm(total = n_files)
    for (dirpath, dirnames, filenames) in os.walk(args.dataset):
        if(len(filenames) == 0): continue   # check if it is empty
        if(filenames[0].endswith(".jpg")==False): continue # check if image is not .jpg extension

        #create label folder in the organized dataset
        label = os.path.basename(os.path.normpath(dirpath))
        train_label_path = os.path.join(train_dataset_path,label)
        val_label_path = os.path.join(val_dataset_path,label)
        test_label_path = os.path.join(test_dataset_path,label)
        if(os.path.exists(train_label_path)==False):  os.mkdir(train_label_path)
        if(os.path.exists(val_label_path)==False):    os.mkdir(val_label_path)
        if(os.path.exists(test_label_path)==False):   os.mkdir(test_label_path)

        filtered_images = []
        for filename in filenames: 
            image = cv2.imread(os.path.join(dirpath,filename)) # read image

            #black image filter
            if(args.black_img_filter): 
                if(not black_img_filter(image)): filtered_images.append(filename)
            else :  filtered_images.append(filename)

            #update progress bar
            progress_bar.update()

        #split training, validation and test images for every label
        random.shuffle(filtered_images)
        training_dataset,val_dataset = train_test_split(filtered_images,test_size=0.3)
        random.shuffle(val_dataset)
        val_dataset,test_dataset = train_test_split(val_dataset,test_size= 0.2)

        for a in training_dataset:
            new_train_path = os.path.join(train_label_path,a)
            old_train_path = os.path.join(dirpath,a)
            shutil.copy(old_train_path,new_train_path)

        for b in val_dataset:     
            new_val_path = os.path.join(val_label_path,b)
            old_val_path = os.path.join(dirpath,b)
            shutil.copy(old_val_path,new_val_path)

        for c in test_dataset:
            new_test_path = os.path.join(test_label_path,c)
            old_test_path = os.path.join(dirpath,c)
            shutil.copy(old_test_path,new_test_path)


    print("done")
#========================================
if __name__ == "__main__": main()