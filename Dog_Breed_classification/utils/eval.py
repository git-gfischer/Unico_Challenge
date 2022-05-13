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
    labels = df['label'].values.tolist()
    scores  = df['pred'].values.tolist()
    times = df['inference_time_seconds']

    label_conv = []
    score_conv = []
    print(labels)

    for i in range(len(labels)):
        label = classes.index(labels[i])
        label_conv.append(label)

        score = classes.index(scores[i])
        score_conv.append(score)
       
    inference_time = times.sum()/len(times)

    print(classification_report(label_conv, score_conv, target_names=classes))
    print(f"Mean inference time: {inference_time} seconds \n")

##====================================================