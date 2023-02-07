import torch
import torch.nn as nn
import pytorch_lightning as pl
from model import Model

class SiameseNetwork(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.loss = torch.nn.BCELoss()
        self.learning_rate = lr
        self.model = Model()
        self.linear = nn.Linear(in_features=128, out_features=1)
        self.activation = nn.Sigmoid()
        self.th = 0.5
        # self.twin_network.apply(SiameseNetwork.init_weights)

    def forward(self, x: (torch.Tensor, torch.Tensor)):
        input1, input2 = x
        batch_size = len(input1)

        cat_input = torch.cat((input1, input2), dim=0)
        cat_output = self.model(cat_input)
        output1 = cat_output[:batch_size]
        output2 = cat_output[batch_size:]

        l1_distance = torch.abs(output1 - output2)

        weighted_distance = self.linear(l1_distance)
        # pred = self.activation(weighted_distance)

        return weighted_distance

    def training_step(self, batch, batch_idx):
        input1, input2, y = batch
        pred = self.forward((input1, input2))
        loss = self.loss(pred.squeeze(), y.float())

        y_pred = (pred > self.th).float().squeeze()
        # y_pred = torch.where(y_pred > self.th, y_pred, 1.).squeeze()
        accuracy = torch.sum(y_pred == y) / len(y_pred)

        self.log("Accuracy", accuracy, on_step=True)
        self.log("Train Loss", loss, on_step=True)

        return loss

    def training_epoch_end(self, outputs):
        ...
        # self.learning_rate = self.learning_rate * 0.99

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=2e-1)
            torch.nn.init.normal_(m.bias, mean=0.5, std=10e-2)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, mean=0, std=10e-2)
            torch.nn.init.normal_(m.bias, mean=0.5, std=10e-2)

    def configure_optimizers(self):
        #TODO: in order to add l2 regularization, add weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer
