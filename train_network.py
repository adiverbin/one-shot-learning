from torch.utils.data import DataLoader
from lfwa_dataset import LFWADataset
import pytorch_lightning as pl
from siamese_network import SiameseNetwork

def train():
    train_dataloader = DataLoader(LFWADataset(data_dir='lfwa/lfw2/lfw2', source_path='lfwa/lfw2/splits/train.txt'),
                            batch_size=128, shuffle=True, num_workers=0)
    trainer = pl.Trainer(max_epochs=200)
    model = SiameseNetwork()
    trainer.fit(model, train_dataloaders=train_dataloader)

if __name__ == '__main__':
    train()