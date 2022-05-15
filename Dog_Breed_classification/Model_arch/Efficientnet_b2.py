from Model_arch.ImageClassificationBase import ImageClassificationBase
from torch import nn
from torchvision import models
import torch
class Efficientnet_b2(ImageClassificationBase):
    def __init__(self, train_flag,n_classes=100):
        super().__init__()
        self.model = models.efficientnet_b2(pretrained=train_flag)
        self.model.classifier[1] = nn.Linear(1408,n_classes)  

        #in_features = self.model.fc.in_features
        #self.model.fc = nn.Sequential(nn.Linear(1408,512),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.2),
        #                                nn.Linear(512, n_classes))

    def forward(self,xd):
        y = self.model(xd)
        #f = self.features(xd)
        #f = f.view(f.size(0), -1)
        #y = self.classifier(f)
        return y