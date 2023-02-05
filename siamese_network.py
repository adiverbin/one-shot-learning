import torch
import torch.nn as nn
import pytorch_lightning as pl


from lfwa_dataset import LFWADataset
from torch.utils.data import DataLoader

class BinaryCrossEntropyLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=4096, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, output1, output2, label):
        l1_distance = torch.abs(output1 - output2)
        weighted_distance = self.linear1(l1_distance)
        distance = self.activation(weighted_distance)

        bceloss = torch.nn.BCELoss()(distance.squeeze(), label.float())
        # loss = (label * torch.log(distance)) + ((1-label) * torch.log(1-distance)) + self.l2_lambda *

        return bceloss


class SiameseNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = BinaryCrossEntropyLoss()
        self.learning_rate = 0.0001
        self.twin_network = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(10, 10)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(7, 7)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(4, 4)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(4, 4)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(4, 4)),

            nn.ReLU(),

            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(4, 4)),

            nn.ReLU(),
            nn.Flatten(),
            # 256X6X6 = 9216
            nn.Linear(in_features=9216, out_features=4096),
            nn.Sigmoid()
        )
        self.twin_network.apply(SiameseNetwork.init_weights)
        self.threshold = 0.5


    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        length = len(input1)
        cat_input = torch.cat((input1, input2), dim=0)
        cat_output = self.twin_network(cat_input)
        output1 = cat_output[:length]
        output2 = cat_output[length:]

        return output1, output2

    def training_step(self, batch, batch_idx):
        input1, input2, y = batch
        output1, output2 = self.forward(input1, input2)
        loss = self.loss.forward(output1, output2, y)
        y_prob = torch.where(loss <= self.threshold, 0, loss)
        y_prob = torch.where(y_prob > self.threshold, 1, y_prob)
        correct = y_prob.eq(y).sum().item()
        total = len(y)
        logs={"train_loss": loss}

        # batch_dictionary = {
        #     "loss": loss,
        #     "log": logs,
        #     "correct": correct,
        #     "total": total
        # }

        self.log("train Loss", loss, on_step=True)
        return loss

    def training_epoch_end(self, outputs):
        print(f"UPDATING LEARNING RATE FROM {self.learning_rate} TO {self.learning_rate * 0.99}")
        self.learning_rate = self.learning_rate * 0.99
    def validation_step(self, batch, batch_idx):
        input1, input2, y = batch
        output1, output2 = self.forward(input1, input2)
        loss = self.loss.forward(output1, output2, y)

        self.log("validation Loss", loss,)
        return loss
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=2e-1)
            torch.nn.init.normal_(m.bias, mean=0.5, std=10e-2)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, mean=0, std=10e-2)
            torch.nn.init.normal_(m.bias, mean=0.5, std=10e-2)
    def test_step(self, batch, batch_idx):
        input1, input2, y = batch
        output1, output2 = self.forward(input1, input2)
        loss = self.loss.forward(output1, output2, y)

        self.log("test Loss", loss, )

    def configure_optimizers(self):
        #TODO: in order to add l2 regularization, add weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

