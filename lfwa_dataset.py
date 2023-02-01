import os

import cv2
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


class LFWADataset(Dataset):
    def __init__(self, data_dir: str, source_path: str):
        super().__init__()
        self.data = {}
        self.dara_dir = data_dir
        with open(source_path) as source_file:
            paris = source_file.readlines()

        for line in enumerate(paris):
            self.data[int(line[0]) - 1] = line[1][:-1].split('\t')

        self.dataset_length = int(self.data[-1][0]) * 2
        del self.data[-1]

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        if len(self.data[item]) == 3:
            label = 1
            first_name = self.data[item][0]
            first_index = self.data[item][1]
            second_name = self.data[item][0]
            second_index = self.data[item][2]
        else:
            label = 0
            first_name = self.data[item][0]
            first_index = self.data[item][1]
            second_name = self.data[item][2]
            second_index = self.data[item][3]

        transform = transforms.Compose([transforms.ToTensor()])

        first_path = f'{first_name}_{first_index.zfill(4)}.jpg'
        second_path = f'{second_name}_{second_index.zfill(4)}.jpg'

        first_image = cv2.imread(os.path.join(self.dara_dir, first_name, first_path))

        second_image = cv2.imread(os.path.join(self.dara_dir, second_name, second_path))

        first_image = transform(first_image)
        second_image = transform(second_image)

        return first_image, second_image, label
