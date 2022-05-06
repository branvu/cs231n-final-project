import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, kernal_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 16, kernal_size=(5, 5))
        self.linear1 = nn.Linear(15 * 5 * 3, 100)
        self.linear2 = nn.Linear(100, 3)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.pool(x)
        x = F.relu(self.c2(x))
        x = F.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
