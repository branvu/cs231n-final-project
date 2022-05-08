from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import cv2
import numpy as np
import torch


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
        sample = {'image': img, 'language': int(
            classifications[0]), 'letter': int(classifications[1])}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
