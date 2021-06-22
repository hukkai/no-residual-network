import torch
import torch.nn as nn
from .layer import SeqBottle




class plainnet(nn.Module):
    def __init__(self, num_layers, num_classes, inter_bias, inter_BN):
        super(plainnet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.MaxPool2d(3, 2, 1))

        self.inter_bias = inter_bias
        self.inter_BN = inter_BN

        self.layer1 = self.__make_layer__(64, 128, num_layers[0])
        self.layer2 = self.__make_layer__(128, 256, num_layers[1], stride=True)
        self.layer3 = self.__make_layer__(256, 512, num_layers[2], stride=True)
        self.layer4 = self.__make_layer__(512, 2048, num_layers[3], stride=True)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes))

    def __make_layer__(self, in_channels, out_channels, num_blocks, stride=False):
        layers = [SeqBottle(in_channels, in_channels, 
                            stride, self.inter_bias, self.inter_BN)]
        for k in range(1, num_blocks):
            out_channel = in_channels if k != num_blocks -1 else out_channels
            layers += [SeqBottle(in_channels, out_channel, 
                                False, self.inter_bias, self.inter_BN)]
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x



def SeqNet50(num_classes=1000, inter_bias=False, inter_BN=False):
    return plainnet([3, 4, 6, 3], num_classes, inter_bias, inter_BN)

def SeqNet101(num_classes=1000, inter_bias=False, inter_BN=False):
    return plainnet([3, 4, 23, 3], num_classes, inter_bias, inter_BN)