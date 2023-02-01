import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pytorch_lightning as pl
import argparse


class BinaryCrossEntropyLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=4096, out_features=1)
        self.activation = nn.Activation('sigmoid')

    def forward(self, output1, output2, label):
        l1_distance = torch.abs(output1 - output2)
        weighted_distance = self.linear1(l1_distance)
        distance = self.activation(weighted_distance)

        bceloss = torch.nn.BCELoss(distance, label)(distance)
        # loss = (label * torch.log(distance)) + ((1-label) * torch.log(1-distance)) + self.l2_lambda *

        return bceloss


class SiameseNetwork(pl.LightningModule):
    def __init__(self, input_shape=(105, 105, 1)):
        super().__init__()
        self.input_shape = input_shape
        self.loss = BinaryCrossEntropyLoss()

        self.twin_network = nn.Sequential(
            nn.Conv2D(in_channels=1,
                      out_channels=64,
                      kernel_size=(10, 10)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2D(in_channels=64,
                      out_channels=128,
                      kernel_size=(7, 7)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2D(in_channels=128,
                      out_channels=128,
                      kernel_size=(4, 4)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2D(in_channels=128,
                      out_channels=256,
                      kernel_size=(4, 4)),

            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=4096),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Dense(1,
                     activation='sigmoid',
                     kernel_initializer=self.initialize_weights,
                     kernel_regularizer=l2(2e-3),
                     bias_initializer=self.initialize_bias)
        )

    def forward(self, input1, input2):
        output1 = self.model(input1)
        output2 = self.model(input2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        input1, input2, y = batch
        output1, output2 = self.forward(input1, input2)
        loss = self.loss.forward(output1, output2, y)

        return loss

    def initialize_weights(self):
        pass

    def initialize_bias(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        #TODO: in order to add l2 regularization, add weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def initialize_conv_weights(shape):
        """
        Initialize the convolution layers weight as the paper says.
        normal distribution with zero-mean and a standard deviation of 10−2.
        :param shape: The shape of the weight in initialzie
        :return: The initialized weights
        """
        return torch.normal(mean=0, std=10e-2, size=shape)






