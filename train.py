from tkinter.filedialog import test
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from mydata import SignLanguageDataset
from models import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 10
epochs = 15

dataset = SignLanguageDataset("annotations.csv", None)
size = len(dataset)
splits = [int(size * 0.80), (size * 0.20), (size * 0.10)]
train_set, val_set, test_set = torch.utils.data.random_split(dataset, splits)

dummy_set = torch.utils.data.random_split(dataset ,[10])

trainLoader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
valLoader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
testLoader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

model = Model()
model.to(device) # Set model to GPU

lo = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(epochs):
    total_loss = 0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        print(labels)
        optimizer.zero_grad()
        out = model(inputs)
        loss = lo(out, labels[0]) # Loss with respect to the 
        loss.backward()
        out.step()
        total_loss += loss.item()
        if i % 10 == 9:
            print(f'epoch {epoch} step {i} loss: {total_loss / 10:.2f}')
            total_loss = 0

print("Done Training")


# Evaluate on validation set
correct = 0
total = 0
with torch.no_grad():
    for data in valLoader:
        images, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = model(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct_occurences = (predicted == labels)
        correct += correct_occurences.sum().item()
print(f"Accuracy on validation set {100 * correct // total} %")

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

classes = {0:'ASL', 1:'JSL', 2:'ISL'}

# get some random training images
dataiter = iter(valLoader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Saving the model
torch.save(model.state_dict(), 'models/model_basic.pth')



