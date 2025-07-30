import torch
import torch.nn as nn
import torchvision.models as models

class Transformer(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_SWAG_LINEAR_V1'):
        super(Transformer, self).__init__()
        # Load pretrained ResNet18
        self.transformer = models.vit_h_14(weights=weights)
        
        in_features = self.transformer.heads.head.in_features
        self.transformer.heads.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.transformer(x)