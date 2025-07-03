import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18_Features(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(ResNet18_Features, self).__init__()
        # Load pretrained ResNet18
        self.resnet18 = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class VGG16_Features(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super().__init__()
        self.vgg16 = models.vgg16(weights=weights)
        self.features = self.vgg16.features  # convolutional part
        self.avgpool = self.vgg16.avgpool    # avgpool layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten to (batch, 25088)
        return x


# To test the model definition:
# if __name__ == "__main__":
#     image = torch.randn(4, 3, 64, 64)

#     model = ResNet18_Features()

#     # input image to model
#     output = model(image)
    