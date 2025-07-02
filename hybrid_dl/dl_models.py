import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(ResNet18, self).__init__()
        # Load pretrained ResNet18
        self.resnet18 = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

# To test the model definition:
if __name__ == "__main__":
    image = torch.randn(4, 3, 64, 64)

    model = ResNet18()

    # input image to model
    output = model(image)
    