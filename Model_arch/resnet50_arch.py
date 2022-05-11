from Model_arch.ImageClassificationBase import ImageClassificationBase
from torch import nn
from torchvision import models

class Resnet50(ImageClassificationBase):
    def __init__(self, train_flag):
        n_classes = 100
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=train_flag)
        #self.features = nn.Sequential(*list(resnet50.children())[:-1])

        # Freeze training for all "features" layers
        for param in self.resnet50.parameters(): param.requires_grad = False

        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, n_classes)

        
        #self.classifier = nn.Sequential(nn.Linear(2048,512),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.2),
        #                                nn.Linear(512, n_classes))
    
    def forward(self,xd):
        #f = self.features(xd)
        #f = f.view(f.size(0), -1)
        #y = self.classifier(f)
        y = self.resnet50(xd)
        return y