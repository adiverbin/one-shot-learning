import torch.nn
from torch import nn
from conv_layer import ConvLayer

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(in_channels=1, out_channels=64, kernel_size=(11, 11), max_pool=True)
        self.conv2 = ConvLayer(in_channels=64, out_channels=128, kernel_size=(7, 7), max_pool=True)
        self.conv3 = ConvLayer(in_channels=128, out_channels=128, kernel_size=(5, 5), max_pool=True)
        self.conv4 = ConvLayer(in_channels=128, out_channels=256, kernel_size=(5, 5), max_pool=True)
        self.conv5 = ConvLayer(in_channels=256, out_channels=256, kernel_size=(5, 5))
        self.conv6 = ConvLayer(in_channels=256, out_channels=256, kernel_size=(5, 5))


        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]

        self.flatten = nn.Flatten()
        self.batchnorm = nn.BatchNorm2d(256)
        self.linear = nn.Linear(in_features=2304, out_features=128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x.data

