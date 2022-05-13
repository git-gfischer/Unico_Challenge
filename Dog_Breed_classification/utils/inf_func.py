#utils inference Functions
import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.transforms import RandomHorizontalFlip
from PIL import Image
import cv2
import numpy as np


def transform_image(img_crop,size_img=224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    my_transforms = Compose ([Resize([size_img, size_img]),
                              ToTensor(),
                              Normalize(mean, std)])
    
    image = Image.fromarray(img_crop)
    return my_transforms(image).unsqueeze(0)
#===========================================
def prediction(model, img_crop, device, threshold=0.5):
    tensor = transform_image(img_crop)
    tensor = tensor.to(device)
    with torch.no_grad():
       output = model.forward(tensor)
       probabilities = torch.nn.Softmax(dim=-1)(output)
    #    sortedProba = torch.argsort(probabilities, dim=-1, descending=False)
       _, sortedProba = torch.max(probabilities, 1)
       #print(torch.mean(output, dim=(1)))
       #score = torch.mean(output, dim=(1))
       #pred = (score > threshold)

    return sortedProba, probabilities
#==========================================
def predict_batch(batch,model):
    with torch.no_grad():
        out = model(batch)
        _, predicted = torch.max(out, 1)
    return predicted
#==========================================
def load_model(network, model_path,device):

    #state = torch.load(model_path,map_location=device)
    #network.load_state_dict(state['state_dict'])
    network.load_state_dict(torch.load(model_path))
    network.eval()
    
    return network
#=========================================
def load_classes(classes_path):
    with open(classes_path, 'r') as classes_file:
        classes = classes_file.readlines()
    return classes
