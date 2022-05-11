import torch
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def calc_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#====================================================
def metrics_csv(csv_path, classes):
    df  = pd.read_csv(csv_path)
    labels = df['label']
    scores  = df['pred']
    times = df['inference_time_seconds']

    label_conv = []
    score_conv = []

    for i in range(len(labels)):
        if (labels[i]=="afro-american"): label_conv.append(0)
        elif (labels[i]=="asian"): label_conv.append(1)
        elif (labels[i]=="caucasian"): label_conv.append(2)
        elif (labels[i]=="indian"): label_conv.append(3)

        if (scores[i]=="afro-american"): score_conv.append(0)
        elif (scores[i]=="asian"): score_conv.append(1)
        elif (scores[i]=="caucasian"): score_conv.append(2)
        elif (scores[i]=="indian"): score_conv.append(3)
    
    inference_time = times.sum()/len(times)

    print(classification_report(label_conv, score_conv, target_names=classes))
    print(f"Mean inference time: {inference_time} seconds \n")

##====================================================