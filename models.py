import torch.nn as nn
import torch.nn.functional as F
import torch


# class ASLModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.c2 = nn.Conv2d(32, 16, kernel_size=(5, 5))
#         self.linear1 = nn.Linear(1296, 100)
#         self.linear2 = nn.Linear(100, 26)
#         self.drop = nn.Dropout(p=0.2)

#     def forward(self, x):
#         x = x.float()
#         x = x.permute(0, 3, 1, 2)
#         x = self.c1(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.drop(F.relu(self.c2(x)))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x

class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.c3 = nn.Conv2d(64, 32, kernel_size=(3, 3))
        self.linear1 = nn.Linear(288, 100)
        self.linear2 = nn.Linear(100, 41)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = self.c1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.c2(x)))
        x = self.drop(F.relu(self.c3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class JSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.c3 = nn.Conv2d(64, 32, kernel_size=(3, 3))
        self.linear1 = nn.Linear(288, 100)
        self.linear2 = nn.Linear(100, 41)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = self.c1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.c2(x)))
        x = self.drop(F.relu(self.c3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ISLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
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


class ARSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 16, kernel_size=(5, 5))
        self.linear1 = nn.Linear(1296, 100)
        self.linear2 = nn.Linear(100, 32)
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


class LanguageModel(nn.Module):
    def __init__(self, num_langs):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 16, kernel_size=(5, 5))
        self.linear1 = nn.Linear(1296, 100)
        self.linear2 = nn.Linear(100, num_langs)

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
