from enum import unique
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import cv2
import numpy as np
import torch

lang_char_counts = {0: 26, 1: 41, 2: 26, 3: 32}


class SignLanguageDataset(Dataset):
    def __init__(self, csv, transform=None):
        self.data = pd.read_csv(csv)
        self.transform = transform
        # print(self.data, len(self.data), self.transform)

    def __getitem__(self, idx):
        # print("get item", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = cv2.imread(self.data.iloc[idx, 0], cv2.IMREAD_GRAYSCALE)
        classifications = np.array(self.data.iloc[idx, 1:])
        total = sum([lang_char_counts[i]
                     for i in range(int(classifications[0]))])
        unique_id = total + int(classifications[1])
        sample = {'image': img, 'language': int(
            classifications[0]), 'letter': int(classifications[1]), 'unique_id': unique_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
