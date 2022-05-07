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
from torch.utils.data.sampler import SubsetRandomSampler


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 10
    epochs = 1

    dataset = SignLanguageDataset("annotations.csv", None)
    size = len(dataset)
    splits = [int(size * 0.80), int(size * 0.10), int(size * 0.10) + 1]
    print("splits", sum(splits), splits, size)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, splits)

    # dummy_set = torch.utils.data.random_split(dataset, [10])

    trainLoader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valLoader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    model = Model()
    model.to(device)  # Set model to GPU

    lo = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start training")
    print("="*50, "\n")

    for epoch in range(epochs):
        total_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(trainLoader, 0):
            inputs, language, letter = data
            inputs = data[inputs].to(device)
            language = data[language].to(device)
            letter = data[letter].to(device)

            optimizer.zero_grad()
            out = model(inputs)

            new_lang = []
            for j in language:
                curr = [0] * 3
                curr[j] = 1
                new_lang.append(curr)
            # print(language, new_lang)
            new_lang = torch.Tensor(new_lang)
            new_lang = new_lang.to(device)
            # language [0, 0, 1]
            # language [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
            loss = lo(out, new_lang)  # Loss with respect to the
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, label = torch.max(out.data, 1)
            total += language.size(0)
            correct_occurences = (label == language)
            correct += correct_occurences.sum().item()

            if i % 200 == 199:
                print(
                    f'epoch {epoch} step {i} loss: {total_loss / 200:.2f} accuracy: {correct  * 100 // total}%')
                total_loss = 0

    print("Done Training")

    # Evaluate on validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valLoader:
            inputs, language, letter = data
            inputs = data[inputs].to(device)
            language = data[language].to(device)
            letter = data[letter].to(device)

            out = model(inputs)
            _, predicted = torch.max(out.data, 1)
            total += language.size(0)
            correct_occurences = (predicted == language)
            correct += correct_occurences.sum().item()
    print(f"Accuracy on validation set {100 * correct // total} %")

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    classes = {0: 'ASL', 1: 'JSL', 2: 'ISL'}

    # # get some random training images
    # dataiter = iter(valLoader)
    # images, language, letter = dataiter.next()

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[language[j]]:5s}' for j in range(batch_size)))

    # Saving the model
    torch.save(model.state_dict(), 'models/model_basic.pth')


if __name__ == '__main__':
    main()
