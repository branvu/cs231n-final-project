import torch.nn as nn
import torch.nn.functional as F
import torch


class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 16, kernel_size=(5, 5))
        self.linear1 = nn.Linear(1296, 100)
        self.linear2 = nn.Linear(100, 26)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = self.c1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop(F.relu(self.c2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 16, kernel_size=(5, 5))
        self.linear1 = nn.Linear(1296, 100)
        self.linear2 = nn.Linear(100, 3)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = self.c1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.relu(self.c2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
