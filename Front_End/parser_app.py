import sys
import os

p = os.path.abspath('../Dog_Breed_classification')
if p not in sys.path: sys.path.append(p)
from Model_arch.resnet50_arch import Resnet50
from utils.utils import to_device
from utils.inf_func import load_model, load_classes
from inference import web_inferece


from flask import Flask
from flask import render_template
from flask import request
import json
import numpy as np
import cv2
import torch


app = Flask(__name__)

model = "../Dog_Breed_classification/experiments/resnet50_SGD_001_LR5/weights/resnet50_dogs_19.pth"
classes_file = "../Dog_Breed_classification/config/labels.txt"

classes = load_classes(classes_file)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = Resnet50(False)
network = to_device(network,device)
network = load_model(network, model, device)

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
        path = request.data
        print(path)
        return json.dumps({'result':'done'}), 200, {'ContentType':'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
