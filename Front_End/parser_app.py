import sys
import os
from Dog_Breed_classification.Train import web_retrain_model

p = os.path.abspath('../Dog_Breed_classification')
if p not in sys.path: sys.path.append(p)
from Model_arch.resnet50_arch import Resnet50
from utils.utils import to_device
from utils.inf_func import load_model, load_classes
from inference import web_inferece
from Enrollment import web_enrollment


from flask import Flask
from flask import render_template
from flask import request
import json
import numpy as np
import cv2
import torch


app = Flask(__name__)

directory = os.getcwd()

#model = "../Dog_Breed_classification/experiments/resnet50_adam_0001_LR5/weights/resnet50_dogs_15.pth"
model_default = os.path.join(directory,"../Dog_Breed_classification/experiments/resnet50_ext_SGD_001_LR5/weights/resnet50_dogs_16.pth")
database = os.path.join(directory,"../dogs_dataset/train_organized")
classes_file = os.path.join(directory,"../Dog_Breed_classification/config/labels.txt")
cfg = os.path.join(directory,"../Dog_Breed_classification/config/dog_rec.yaml")

classes = load_classes(classes_file)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = Resnet50(False)
network = to_device(network,device)
network = load_model(network, model_default, device)

@app.route('/', methods = ['POST','GET'])
def index():
    if request.method == 'POST':
        img_bin = request.files['file'].read() # image binary
        arr = np.fromstring(img_bin, np.uint8)
        image_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # read JPG
        result,prob = web_inferece(image_np,network,device,classes)
        return json.dumps({'result':result,'prob':prob}), 200, {'ContentType':'application/json'}
       #return result, 200, {'ContentType':'application/json'}
    else:
        return render_template('index.html')

@app.route('/enroll',methods = ['POST','GET'])
def enroll_app():
    if request.method == 'POST':
        path = request.data.decode("utf-8")
        web_enrollment(path,database,classes_file) # enroll images on dataset
        new_model = web_retrain_model(cfg)         # retrain model with new labels
        network = load_model(network, new_model, device)
        return json.dumps({'result':'done'}), 200, {'ContentType':'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
