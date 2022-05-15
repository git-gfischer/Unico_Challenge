from Model_arch.ImageClassificationBase import ImageClassificationBase
from torch import nn
from torchvision import models

class Resnet18(ImageClassificationBase):
    def __init__(self, train_flag,n_classes=100):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=train_flag)

        # Freeze training for all "features" layers
        for param in self.resnet18.parameters(): param.requires_grad = False

        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, n_classes)

        #self.resnet18.fc = nn.Sequential(nn.Linear(in_features,512),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.2),
        #                                nn.Linear(512, n_classes))

        #self.features = nn.Sequential(*list(resnet18.children())[:-1])
        #self.classifier = nn.Sequential(nn.Linear(512,n_classes))
    
    def forward(self,xd):
        #f = self.features(xd)
        #f = f.view(f.size(0), -1)
        #y = self.classifier(f)
        y = self.resnet18(xd)
        return y