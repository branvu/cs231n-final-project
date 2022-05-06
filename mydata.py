from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import cv2
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, csv, transform=None):
        self.data = pd.read_csv(csv)
        self.transform = transform
    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx, 0])
        classifications = np.array(self.data[idx, 1:])
        sample = {'image':img, 'language':int(classifications[0]), 'letter': int(classifications[1])}

        if self.transform:
            sample = self.transform(sample)

        return sample


