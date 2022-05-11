from Model_arch.ImageClassificationBase import ImageClassificationBase
from torch import nn
from torchvision import models

class Resnet18(ImageClassificationBase):
    def __init__(self, train_flag):
        n_classes = 100
        super().__init__()
        resnet18 = models.resnet18(pretrained=train_flag)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(512,n_classes))
    
    def forward(self,xd):
        f = self.features(xd)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y